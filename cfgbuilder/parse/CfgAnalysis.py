import re
from parse.Cfg import Cfg
from parse.BlockType import BlockType
from parse.BasicBlock import BasicBlock
from parse.entity import OrderDict, Triplet, Stack, Logger


class CfgAnalysis:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.basicblocks = cfg.basicBlocks
        self.fliter_fallback = False
        self.fliter_dispatcher = False
        self.fliter_function = False
        self._fliterset = set()

        self.STATE_CHANGE_OPCODES = {'SSTORE', 'LOG0', 'LOG1', 'LOG2', 'LOG3', 'LOG4', 'CREATE', 'CALL',
                                     'SELFDESTRUCT'}

        self.predecessors = {}

    def _enhance_node_types(self):
        for offset, block in self.basicblocks:
            opnames = {inst.name for inst in block.instructions}

            if 'SELFDESTRUCT' in opnames:
                block.setType(BlockType.SUICIDE_BLOCK)
            elif 'REVERT' in opnames or 'INVALID' in opnames:
                block.setType(BlockType.TERMINAL_FAILURE)

            elif block.type in [BlockType.COMMON, BlockType.UNDEFINED]:
                if self.STATE_CHANGE_OPCODES.intersection(opnames):
                    block.setType(BlockType.STATE_CHANGE)
                elif 'CALLVALUE' in opnames:
                    block.setType(BlockType.MONEY_IN)

    def _enhance_edge_types(self):
        for offset, block in self.basicblocks:
            if not block.hasSuccessor():
                continue

            last_op = block.end

            if not hasattr(block, '_typed_edges'):
                block._typed_edges = []

            if last_op.name == 'JUMPI':
                fallthrough_target = last_op.pc + 1
                for suc in block.successors:
                    if suc.offset == fallthrough_target:
                        block._typed_edges.append({'target': suc.offset, 'type': 'conditional_false'})
                    else:
                        block._typed_edges.append({'target': suc.offset, 'type': 'conditional_true'})
            elif last_op.name == 'JUMP':
                if block.successors:
                    block._typed_edges.append({'target': block.successors[0].offset, 'type': 'unconditional'})
            elif not block.is_terminator:
                if block.successors:
                    block._typed_edges.append({'target': block.successors[0].offset, 'type': 'sequential'})

    def _build_predecessors(self):
        self.predecessors = {off.key: [] for off in self.basicblocks}
        for offset, block in self.basicblocks:
            for suc in block.successors:
                if suc.offset in self.predecessors:
                    self.predecessors[suc.offset].append(offset)

    def _mark_pure_guard_paths(self):
        terminal_failure_blocks = {off.key for off in self.basicblocks
                                   if off.value.type == BlockType.TERMINAL_FAILURE}

        nodes_to_prune = set()
        worklist = list(terminal_failure_blocks)
        visited = set(terminal_failure_blocks)

        while worklist:
            offset = worklist.pop(0)
            block: BasicBlock = self.basicblocks.get(offset)
            if not block: continue

            opcodes = {inst.name for inst in block.instructions}

            has_state_change = len(opcodes.intersection(self.STATE_CHANGE_OPCODES)) > 0

            can_be_pruned = not has_state_change

            if can_be_pruned:
                nodes_to_prune.add(offset)

                for pred_offset in self.predecessors.get(offset, []):
                    if pred_offset not in visited:
                        pred_block: BasicBlock = self.basicblocks.get(pred_offset)
                        if pred_block:

                            all_succs_are_prunable = True
                            for succ in pred_block.successors:

                                if succ.offset != -1 and \
                                        succ.offset not in nodes_to_prune and \
                                        succ.offset not in terminal_failure_blocks:
                                    all_succs_are_prunable = False
                                    break

                            if all_succs_are_prunable:
                                worklist.append(pred_offset)
                                visited.add(pred_offset)

        self._fliterset.update(nodes_to_prune)

    def analyse(self):
        self._enhance_node_types()
        self._enhance_edge_types()
        self._build_predecessors()

        self.check_fallback()
        self.check_function()

        self._mark_pure_guard_paths()

        self.filter()

    def check_fallback(self):
        for offset in self.cfg.fallbacks:
            block: BasicBlock = self.basicblocks.get(offset)

            if block.has_caller() and not self.is_zero_fallback(block):
                return
        self.fliter_fallback = True

    def check_function(self):
        self.fliter_dispatcher = True
        allset = set()
        retainset = set()
        for offset in self.cfg.functions:
            call = False
            queue = Stack()
            visited = set()
            queue.push(offset)
            while not queue.isEmpty():
                eoffset = queue.pop()
                visited.add(eoffset)
                block: BasicBlock = self.basicblocks.get(eoffset)

                if not block:
                    continue

                if block.has_caller() and not self.is_zero_fallback(block):
                    call = True
                for i in block.successors:
                    if (i.offset not in visited) and (i.offset not in self.cfg.dispatchers):
                        queue.push(i.offset)
            if call:
                retainset.update(visited)
            allset.update(visited)

        original_fliterset = allset - retainset
        self._fliterset.update(original_fliterset)

        if len(original_fliterset) / self.cfg.length < 0.5:
            self.fliter_function = True

    def filter(self):
        if self.fliter_fallback:
            for offset in self.cfg.fallbacks:
                block: BasicBlock = self.basicblocks.get(offset)
                if block: block.setretain(False)

        if self.fliter_dispatcher:
            for offset in self.cfg.dispatchers:
                block: BasicBlock = self.basicblocks.get(offset)
                if block: block.setretain(False)

        if self._fliterset:
            for offset in self._fliterset:
                block: BasicBlock = self.basicblocks.get(offset)
                if block:
                    block.setretain(False)

    def is_zero_fallback(self, block: BasicBlock):
        if block.length >= 5 and block.getInstruction(-1).name == "JUMPI" and \
                "PUSH" in block.getInstruction(-2).name and \
                block.getInstruction(-3).name == "ISZERO" and \
                block.getInstruction(-4).name == "CALLVALUE" and \
                block.getInstruction(-5).name == "JUMPDEST":
            return True
        return False

    def extract_list(self):
        contract_list = []
        blocks = self.basicblocks
        removedig = re.compile(r'[0-9]+')
        for pair in blocks:
            block: BasicBlock = pair.value
            if block.retain:
                ins = block.instructions
                blocklist = []
                for i in ins:
                    instr = i.name
                    word = removedig.sub('', instr)
                    if word != "":
                        blocklist.append(word)
                if blocklist != []:
                    contract_list.append(blocklist)
        return contract_list

    def analyse_icws(self):
        self.check_fallback()
        self.fliter_dispatcher = True
        self.filter()
