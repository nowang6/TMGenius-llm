# -*- coding: utf-8 -*-
# Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections import deque, defaultdict
from itertools import chain
from typing import Deque, List, Dict, Sequence, Set, Tuple, Union, Optional, Type, TypeVar

import numpy as np

from auto_optimizer.graph_refactor.interface.base_node import PlaceHolder, Initializer, Node

NodeTypeVar = TypeVar('NodeTypeVar', PlaceHolder, Initializer, Node)
NodeType = Union[PlaceHolder, Initializer, Node]


class NodeNotExistException(KeyError):
    def __init__(self, node_name: str) -> None:
        super().__init__()
        self.value = '{} is not exist in graph'.format(node_name)

    def __str__(self) -> str:
        return repr(self.value)


class BaseGraph(ABC):

    def __init__(
        self,
        name: str,
        nodes: Optional[List[Node]] = None,
        inputs: Optional[List[PlaceHolder]] = None,
        outputs: Optional[List[PlaceHolder]] = None,
        initializers: Optional[List[Initializer]] = None,
        value_infos: Optional[List[PlaceHolder]] = None,
    ) -> None:
        super().__init__()
        self.name: str = name
        self._nodes: List[Node] = [] if nodes is None else nodes
        self._inputs: List[PlaceHolder] = [] if inputs is None else inputs
        self._outputs: List[PlaceHolder] = [] if outputs is None else outputs
        self._initializers: List[Initializer] = [] if initializers is None else initializers
        self._value_infos: List[PlaceHolder] = [] if value_infos is None else value_infos

        self._node_map: Dict[str, NodeType] = {}
        self._value_map: Dict[str, PlaceHolder] = {}
        self._prev_map: Dict[str, Node] = {}
        self._next_map: Dict[str, List[Node]] = defaultdict(list)

        self.update_map()

    def __getitem__(self, key: str) -> NodeType:
        if not self._node_map.get(key, None):
            raise KeyError("node '{}' not in graph!".format(key))
        return self._node_map.get(key)

    def __setitem__(self, key: str, value: NodeType) -> None:
        src_node = self._node_map.pop(key, None)
        if not src_node:
            raise KeyError("You are trying to replace node '{}', which does not exist!".format(key))

        if isinstance(src_node, Node):
            self._nodes.remove(src_node)
            # op -> op
            if isinstance(value, Node):
                value.inputs = src_node.inputs
                value.outputs = src_node.outputs
                for i in src_node.inputs:
                    self._next_map.get(i).append(value)
                    self._next_map.get(i).remove(src_node)
                for opt in src_node.outputs:
                    self._prev_map[opt] = value
            # op -> input
            elif value in self._inputs:
                for i in src_node.inputs:
                    self._next_map.get(i).remove(src_node)
                opt = src_node.outputs[0]
                for n in self.get_next_nodes(opt):
                    n.inputs[n.get_input_id(opt)] = value.name
                    self._next_map[value.name] = [n]
                self._prev_map.pop(opt, None)
                self._next_map.pop(opt, None)
        elif isinstance(src_node, Initializer):
            self._initializers.remove(src_node)
            # ini -> input
            if value in self._inputs:
                for n in self.get_next_nodes(src_node.name):
                    n.inputs[n.get_input_id(src_node.name)] = value.name
                    self._next_map[value.name] = [n]
                self._next_map.pop(src_node.name, None)
        else:
            raise RuntimeError("Unsupported!")

    @property
    def inputs(self) -> List[PlaceHolder]:
        return self._inputs

    @property
    def outputs(self) -> List[PlaceHolder]:
        return self._outputs

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @property
    def initializers(self) -> List[Initializer]:
        return self._initializers

    @initializers.setter
    def initializers(self, value):
        self._initializers = value

    @property
    def value_infos(self) -> List[PlaceHolder]:
        return self._value_infos

    @property
    def node_map(self):
        return self._node_map

    @property
    def next_map(self):
        return self._next_map

    @property
    def prev_map(self):
        return self._prev_map

    @classmethod
    @abstractmethod
    def parse(cls, model) -> 'BaseGraph':
        raise NotImplementedError()

    @abstractmethod
    def add_input(self, name: str, dtype: np.dtype, shape: Sequence[Union[int, str]]) -> PlaceHolder:
        raise NotImplementedError()

    @abstractmethod
    def add_output(self, name: str, dtype: np.dtype, shape: Sequence[Union[int, str]]) -> PlaceHolder:
        raise NotImplementedError()

    @abstractmethod
    def add_initializer(self, name: str, value: np.ndarray) -> Initializer:
        raise NotImplementedError()

    @abstractmethod
    def add_node(
        self,
        name: str,
        op_type: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        attrs: Optional[Dict[str, object]] = None,
        domain: str = '',
    ) -> Node:
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def extract(
        self,
        new_model_save_path: str,
        input_name_list: List[str],
        output_name_list: List[str],
        enable_model_check: bool = True,
    ) -> 'BaseGraph':
        raise NotImplementedError()

    @abstractmethod
    def simplify(self, **kwargs) -> 'BaseGraph':
        raise NotImplementedError()

    @abstractmethod
    def infer_shape(self) -> None:
        raise NotImplementedError()

    def update_map(self) -> None:
        # clear map first
        self._node_map.clear()
        self._value_map.clear()
        self._prev_map.clear()
        self._next_map.clear()

        for n in chain(self._inputs, self._outputs, self._nodes, self._initializers):
            _init = self._node_map.get(n.name, None)
            if _init is not None:
                # Initializer and PlaceHolder have same name
                if isinstance(n, Initializer) and isinstance(_init, PlaceHolder):
                    if self._node_map.get(n.name) in self._inputs:
                        self._inputs.remove(_init)
                    else:
                        self._outputs.remove(_init)
                # Node and PlaceHolder have same name
                else:
                    raise RuntimeError(
                        "Duplicate names! {}: '{}' and {}: '{}' have same name,\
                        please use add_name_suffix=True in graph parse".format(
                            type(n).__name__, n.name, type(self._node_map.get(n.name)).__name__, n.name
                        )
                    )
            self._node_map[n.name] = n

        self._value_map = {v.name: v for v in self._value_infos}

        for n in self._nodes:
            # update prev node info
            for opt in n.outputs:
                # if output name not in map
                if not self._prev_map.get(opt):
                    self._prev_map[opt] = n
                    pass
            # update next node info
            for i in set(n.inputs):
                if not self._next_map.get(i):
                    self._next_map[i] = [n]
                else:
                    self._next_map.get(i).append(n)

    def insert_node(self, refer_name: str, insert_node: Node, refer_index: int = 0, mode: str = 'after') -> None:
        """Insert a node with single input and output

        Args:
            refer_name: reference node
            insert_node: the node to be inserted
            refer_index: specifies the inserting position within reference node's output id when mode='after';
                         specifies the inserting position within reference node's input id when mode='before';
                         Default 0.
            mode: insert the node before or after the reference node. Default 'after'.
        """
        # single input and output
        # the value for mode argument

        if refer_name not in self._node_map.keys():
            raise KeyError(f'The node name"{refer_name}" not exists in graph')
        else:
            refer_node = self._node_map.get(refer_name)

        # reference node is input or initializer: convert to inserting node before the next node
        input_flag = False
        if isinstance(refer_node, Initializer) or refer_node in self._inputs:
            if mode == 'before':
                raise RuntimeError(f'Can not insert node before {refer_node.name}.')
            name = refer_node.name
            refer_node = self._next_map.get(name)[0]
            refer_index = refer_node.inputs.index(name)
            mode = 'before'
            input_flag = True

        # reference node is output: convert to inserting node after the prev node
        output_flag = False
        if refer_node in self._outputs:
            if mode == 'after':
                raise RuntimeError(f'Can not insert node after {refer_node.name}.')
            name = refer_node.name
            refer_node = self._prev_map.get(name)
            refer_index = refer_node.outputs.index(name)
            mode = 'after'
            output_flag = True

        # reference node is operator node
        if mode == 'after':
            refer_out_name = refer_node.outputs[refer_index]
            new_out_name = f'{refer_node.name}/{insert_node.name}'
            # connect insert node
            refer_node.outputs[refer_index] = new_out_name
            insert_node.inputs = [new_out_name]
            insert_node.outputs = [refer_out_name]
            # update prev and next map
            self._prev_map[new_out_name] = refer_node
            self._next_map[new_out_name] = [insert_node]
            self._prev_map[refer_out_name] = insert_node
            # deal with situation of inserting node before output
            if output_flag and self._next_map.get(refer_out_name):
                for node in self._next_map.get(refer_out_name):
                    index = node.get_input_id(refer_out_name)
                    node.inputs[index] = new_out_name
                    self._next_map.get(new_out_name).append(node)
                    self._next_map[refer_out_name] = []
        elif mode == 'before':
            refer_in_name = refer_node.inputs[refer_index]
            new_in_name = f'{insert_node.name}/{refer_node.name}'
            # connect insert node
            refer_node.inputs[refer_index] = new_in_name
            insert_node.inputs = [refer_in_name]
            insert_node.outputs = [new_in_name]
            # update prev and next map
            self._prev_map[new_in_name] = insert_node
            self._next_map[new_in_name] = [refer_node]
            self._next_map.get(refer_in_name).append(insert_node)
            self._next_map.get(refer_in_name).remove(refer_node)
            if input_flag:
                # deal with situation of inserting node before output
                for node in self._next_map.get(refer_in_name):
                    if node.name != insert_node.name:
                        index = node.get_input_id(refer_in_name)
                        node.inputs[index] = new_in_name
                        self._next_map.get(new_in_name).append(node)
                self._next_map[refer_in_name] = [insert_node]

        self._node_map[insert_node.name] = insert_node

    def connect_node(self, insert_node: Node, prev_nodes_info: List[str], next_nodes_info: List[str]) -> None:
        """Insert a node with multiple inputs and outputs,
        connect the input and output of insert_node automatically.

        Example:
            g.connect_node(
                Split_0,
                ['Add_0:0', 'split_ini'],
                ['Transpose_0', 'Transpose_1', 'Transpose_2']
            )
        """
        # create inputs and outputs of insert_node
        insert_node.inputs = [f'{insert_node.name}_in_{idx}' for idx in range(len(prev_nodes_info))]
        insert_node.outputs = [f'{insert_node.name}_out_{idx}' for idx in range(len(next_nodes_info))]

        # parse information of previous and next nodes
        prev_info_list, next_info_list = self._parse_nodes_info(prev_nodes_info, next_nodes_info)

        # check if next nodes include the graph output
        output_flag = False
        for node_info in next_info_list:
            if self._node_map.get(node_info['next_node_name']) in self._outputs:
                output_flag = True

        # connect the insert node to previous nodes
        for node_info in prev_info_list:
            node = self._node_map.get(node_info['prev_node_name'])
            out_idx = node_info['prev_node_output_idx']
            in_idx = node_info['insert_node_input_idx']
            if node in self._outputs:
                # the prev node is graph output: illegal
                raise RuntimeError('Please check the list of prev_nodes_info.')
            elif isinstance(node, Initializer) or node in self._inputs:
                # the prev node is graph input or init
                self._set_input_of_node(insert_node, node.name, input_index=in_idx)
            elif node.outputs[out_idx] in [opt.name for opt in self._outputs] and output_flag:
                # the prev node is neighbor of graph output
                self._set_output_of_node(node, insert_node.inputs[in_idx], output_index=out_idx, next_node=insert_node)
            else:
                # the prev node is operator:
                self._set_input_of_node(insert_node, node.outputs[out_idx], input_index=in_idx)

        # connect the insert node to next nodes
        for node_info in next_info_list:
            node = self._node_map.get(node_info['next_node_name'])
            out_idx = node_info['insert_node_output_idx']
            in_idx = node_info['next_node_input_idx']
            if node in self._inputs:
                # the next node is graph input: illegal
                raise RuntimeError('Please check the list of next_nodes_info.')
            elif node in self._outputs:
                # the next node is graph output
                self._set_output_of_node(insert_node, node.name, output_index=out_idx)
            else:
                # the next node is operator
                self._set_input_of_node(node, insert_node.outputs[out_idx], input_index=in_idx, prev_node=insert_node)

    def get_node(self, name: str, node_type: Type[NodeTypeVar]) -> Optional[NodeTypeVar]:
        """Return node with specificed type and name.
        If the node does not exist, return None.
        """
        # try get initializer/operator/input/output from _node_map
        node: Optional[NodeType] = self._node_map.get(name)
        if node and isinstance(node, node_type):
            return node
        # fallback to value_info from _value_map
        if node_type == PlaceHolder:
            ph = self._value_map.get(name)
            if ph and isinstance(ph, node_type):
                return ph
        return None

    def get_nodes(self, op_type: str) -> List[NodeType]:
        nodes = [node for node in self._node_map.values() if node.op_type == op_type]
        return nodes

    def get_value_info(self, io_name: str) -> PlaceHolder:
        return self._value_map.get(io_name)

    def remove(self, name: str, mapping: Optional[Dict[int, int]] = None) -> bool:
        """Remove a specific node from graph
        If map is not provided, it will simply connect the previous node of first input and next nodes of first output.

        Args:
            name: name of the node to be removed
            maps: auto connection map, Default = {0:0}. Keys should be input ids of current node and values should
                  be output ids of current node.

        Return:
            True if remove succeeds, otherwise False
        """
        maps: Dict[int, int] = {0: 0} if mapping is None else mapping
        node = self._node_map.get(name, None)
        if not node:
            raise KeyError("You are trying to remove node '{}', which does not exist!".format(name))
        self._node_map.pop(name, None)
        if node in self._inputs:
            self._inputs.remove(node)
            self._next_map.pop(name, None)
            return True
        if node in self._outputs:
            self._outputs.remove(node)
            self._prev_map.pop(name, None)
            return True
        if isinstance(node, Initializer):
            self._initializers.remove(node)
            self._next_map.pop(name, None)
            return True
        if isinstance(node, Node):
            self._nodes.remove(node)
            for in_id, in_name in enumerate(node.inputs):
                # update next map, node is no longer a next node
                if self._next_map.get(in_name, None):
                    if node in self._next_map.get(in_name):
                        self._next_map.get(in_name).remove(node)
                    if not self._next_map.get(in_name):
                        self._next_map.pop(in_name, None)
                out_id = maps.get(in_id, None)
                # out_id exists, do connection
                if out_id is not None:
                    out_name = node.outputs[out_id]
                    next_nodes = self.get_next_nodes(out_name)
                    if next_nodes:
                        for next_node in next_nodes:
                            for next_node_in_id in next_node.get_input_ids(out_name):
                                next_node.inputs[next_node_in_id] = in_name
                            # update next map, prev node has new next node
                            if self._next_map.get(in_name) is None:
                                self._next_map[in_name] = [next_node]
                            else:
                                self._next_map.get(in_name).append(next_node)
                    elif self._node_map.get(out_name, None):
                        # prev_node -> node -> placeholder
                        prev_node = self._prev_map.get(in_name, None)
                        if prev_node:
                            for prev_next_node in self._next_map.get(in_name, []):
                                prev_next_node_in_id = prev_next_node.get_input_id(in_name)
                                prev_next_node.inputs[prev_next_node_in_id] = out_name
                            prev_node.outputs[prev_node.get_output_id(in_name)] = out_name
                            node.outputs.remove(out_name)
            # update prev and next map, outputs of node no long exist
            for out_name in node.outputs:
                self._prev_map.pop(out_name, None)
                self._next_map.pop(out_name, None)
            return True
        return False

    def get_prev_node(self, input_name: str) -> Optional[Node]:
        return self._prev_map.get(input_name, None)

    def get_next_nodes(self, output_name: str) -> List[Node]:
        return self._next_map.get(output_name, [])

    def toposort(self) -> None:
        def visited_all_prev_nodes(node: Node, visited: Set[Node]) -> bool:
            for input_name in node.inputs:
                prev_node = self.get_prev_node(input_name)
                if prev_node not in visited and prev_node:
                    return False
            return True

        self.update_map()

        queue: Deque[Node] = deque()
        visited: Set[Node] = set()
        for node in self._nodes:
            if visited_all_prev_nodes(node, visited):
                queue.append(node)

        sorted_nodes: List[Node] = []
        while queue:
            node = queue.popleft()
            if visited_all_prev_nodes(node, visited):
                sorted_nodes.append(node)
                visited.add(node)
                for output_name in node.outputs:
                    for next_node in self.get_next_nodes(output_name):
                        if (
                            next_node not in queue
                            and next_node not in visited
                            and visited_all_prev_nodes(next_node, visited)
                        ):
                            queue.append(next_node)

        if len(self._nodes) != len(sorted_nodes):
            raise RuntimeError('Cycle detected in graph!')
        else:
            self._nodes = sorted_nodes

    def remove_unused_nodes(self) -> None:
        self.update_map()

        # Initialize out_degree dict
        out_degree: Dict[Node, int] = dict()
        for node in self._nodes:
            out_degree[node] = 0
            next_nodes = []
            for output_name in node.outputs:
                if output_name in [output.name for output in self.outputs]:
                    next_nodes.append(self._node_map.get(output_name))
                next_nodes.extend(self.get_next_nodes(output_name))
            out_degree[node] = len(set(next_nodes))

        # remove unused operator nodes
        removed: Set[Node] = set()
        queue: Deque[Node] = deque([n for n in out_degree.keys() if out_degree.get(n) == 0])
        while queue:
            node = queue.popleft()
            self._nodes.remove(node)
            removed.add(node)
            for input_name in node.inputs:
                prev_node = self.get_prev_node(input_name)
                if prev_node and prev_node not in queue and prev_node not in removed:
                    out_degree[prev_node] -= 1
                    if out_degree.get(prev_node) == 0:
                        queue.append(prev_node)

        # remove unused graph inputs and initializers
        inputs = [
            inp 
            for n in self._nodes 
            for inp in n.inputs
        ]
        self._inputs = list(filter(lambda x: x.name in inputs, self._inputs))
        self._initializers = list(filter(lambda x: x.name in inputs, self._initializers))

        self.update_map()

    def _add_input(self, graph_input: PlaceHolder) -> PlaceHolder:
        if self._node_map.get(graph_input.name, None):
            raise ValueError("node name '{}' already exists!".format(graph_input.name))
        self._node_map[graph_input.name] = graph_input
        self._inputs.append(graph_input)
        return graph_input

    def _add_output(self, graph_output: PlaceHolder) -> PlaceHolder:
        if self._node_map.get(graph_output.name, None):
            raise ValueError("node name '{}' already exists!".format(graph_output.name))
        self._node_map[graph_output.name] = graph_output
        self._outputs.append(graph_output)
        return graph_output

    def _add_initializer(self, initializer: Initializer) -> Initializer:
        if self._node_map.get(initializer.name, None):
            raise ValueError("node name '{}' already exists!".format(initializer.name))
        self._node_map[initializer.name] = initializer
        self._initializers.append(initializer)
        return initializer

    def _add_node(self, node: Node) -> Node:
        if self._node_map.get(node.name, None):
            raise ValueError("node name '{}' already exists!".format(node.name))
        self._node_map[node.name] = node
        self._nodes.append(node)
        return node

    def _parse_nodes_info(
        self, prev_nodes_info: List[str], next_nodes_info: List[str]
    ) -> Tuple[List[Dict[str, Union[int, str]]], List[Dict[str, Union[int, str]]]]:
        """Parse information of prev_nodes_info and next_nodes_info

        Example:
            prev_info_list = [
                {'prev_node_name': 'Add_0', 'prev_node_output_idx': 0},
                {'prev_node_name': 'split_ini', 'prev_node_output_idx': 0}
                ]
            next_info_list = [
                {'next_node_name': 'Transpose_0', 'next_node_input_idx': 0, 'insert_node_output_idx': 0},
                {'next_node_name': 'Transpose_1', 'next_node_input_idx': 0, 'insert_node_output_idx': 1},
                {'next_node_name': 'Transpose_2', 'next_node_input_idx': 0, 'insert_node_output_idx': 2}
                ]
        """
        prev_info_list: List[Dict[str, Union[int, str]]] = []
        next_info_list: List[Dict[str, Union[int, str]]] = []

        for idx, node_info in enumerate(prev_nodes_info):
            info: Dict[str, Union[int, str]] = {}
            info_splits = node_info.split(':')
            info['prev_node_name'] = info_splits[0]
            if len(info_splits) == 1:
                info['prev_node_output_idx'] = 0
            else:
                info['prev_node_output_idx'] = int(info_splits[1])
            info['insert_node_input_idx'] = idx
            prev_info_list.append(info)

        for idx, str_info in enumerate(next_nodes_info):
            nodes_info = str_info.split(';')
            if len(nodes_info) > 1:
                for i, node_info in enumerate(nodes_info):
                    next_node_name = node_info.split(':')[0]
                    if self._node_map.get(next_node_name) in self._outputs:
                        # when next node share the same edge with graph output, deal with graph output first
                        nodes_info[0], nodes_info[i] = nodes_info[i], nodes_info[0]
            for node_info in nodes_info:
                info_splits = node_info.split(':')
                if len(info_splits) == 1:
                    info = {}
                    info['next_node_name'] = info_splits[0]
                    info['next_node_input_idx'] = 0
                    info['insert_node_output_idx'] = idx
                    next_info_list.append(info)
                else:
                    for elem in info_splits[1].split(','):
                        info = {}
                        info['next_node_name'] = info_splits[0]
                        info['next_node_input_idx'] = int(elem)
                        info['insert_node_output_idx'] = idx
                        next_info_list.append(info)
        return prev_info_list, next_info_list

    def _set_input_of_node(self, node, new_input_name, input_index, prev_node=None) -> None:
        """Change one input edge of the node"""
        old_input_name = node.inputs[input_index]
        node.inputs[input_index] = new_input_name
        # update map for old_input
        if self.get_next_nodes(old_input_name) and node in self._next_map.get(old_input_name):
            self._next_map.get(old_input_name).remove(node)
        # update map for new_input
        if not self.get_next_nodes(new_input_name):
            self._next_map[new_input_name] = [node]
        elif node not in self._next_map.get(new_input_name):
            self._next_map.get(new_input_name).append(node)
        if prev_node is not None:
            self._prev_map[new_input_name] = prev_node

    def _set_output_of_node(self, node, new_output_name, output_index, next_node=None) -> None:
        """Change one output edge of the node"""
        old_output_name = node.outputs[output_index]
        node.outputs[output_index] = new_output_name
        # update map for new_output
        self._prev_map[new_output_name] = node
        for n in self.get_next_nodes(old_output_name):
            index = n.get_input_id(old_output_name)
            n.inputs[index] = new_output_name
            if not self.get_next_nodes(new_output_name):
                self._next_map[new_output_name] = [n]
            else:
                self._next_map.get(new_output_name).append(n)
        if next_node is not None:
            self._next_map[new_output_name] = [next_node]
        # update map for old_output
        self._prev_map.pop(old_output_name, None)
        self._next_map.pop(old_output_name, None)
