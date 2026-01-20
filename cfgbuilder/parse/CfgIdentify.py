from parse.Cfg import Cfg
from parse.BlockType import BlockType
from parse.BasicBlock import BasicBlock
from parse.entity import OrderDict,Triplet,Stack,Logger
import re
BLOCK_LENGTH = 100
# 对cfg中的所有内容进行识别
# 里面的函数会直接对cfg进行改动
class CfgIdentify:
    def __init__(self):
        pass

    def identify(self,cfg:Cfg):
        if cfg.basicblocks.length <= BLOCK_LENGTH:
            cfg.tooshort()
        # 识别dispatcher
        self.finddispatcher(cfg)
        # 找到每一个函数的起始块
        self.findfunction(cfg)
        # 将每个基本块标记其类型
        self.markblocks(cfg)
        return cfg

    def finddispatcher(self,cfg:Cfg):
        for basicblock in cfg.basicblocks:
            offset:int = basicblock.key
            block:BasicBlock = basicblock.value
            if block.length >= 5:
                if self.isDispatcher(block):
                    block.setType(BlockType.DISPATCHER)
                    cfg.add_dispatcher(offset)
    
    def findfunction(self,cfg:Cfg):
        if cfg.dispatcher_num > 0:
            for offset in cfg.dispatchers:
                block:BasicBlock = cfg.basicblocks.get(offset)
                if block:
                    nextoffset = block.end.pc + 1
                if len(block.successors) == 2:
                    for suc in block.successors:
                        if suc.offset > nextoffset:
                            cfg.add_function(suc.offset)
    
    def markblocks(self,cfg:Cfg):
        if cfg.dispatcher_num == 0:
            for basicblock in cfg.basicblocks:
                offset:int = basicblock.key
                block:BasicBlock = basicblock.value
                if block.type == BlockType.UNDEFINED:
                    if offset == 0x0:
                        block.setType(BlockType.START)
                    else:
                        block.setType(BlockType.COMMON)
                    block.checkcaller()
        else:
            firstdispatcher = min(cfg.dispatchers)
            lastdispatcher = max(cfg.dispatchers)
            firstfunctionoffset = min(cfg.functions)
            for basicblock in cfg.basicblocks:
                offset:int = basicblock.key
                block:BasicBlock = basicblock.value
                if block.type == BlockType.UNDEFINED:
                    if offset == 0x0:
                        block.setType(BlockType.START)
                    elif offset > 0x0 and offset < firstdispatcher:
                        block.setType(BlockType.LOADER)
                        cfg.add_loader(offset)
                    elif offset > lastdispatcher and offset < firstfunctionoffset:
                        block.setType(BlockType.FALLBACK)
                        cfg.add_fallback(offset)
                    elif offset >= firstfunctionoffset:
                        block.setType(BlockType.COMMON)
                    block.checkcaller()


    def isDispatcher(self,block:BasicBlock):
        if block.getInstruction(-5).group == "Duplication Operations" and \
                    block.getInstruction(-4).group == "Push Operations" and \
                    block.getInstruction(-3).name == "EQ" and \
                    block.getInstruction(-2).group == "Push Operations" and\
                    block.getInstruction(-1).name == "JUMPI":
            return True
        elif block.getInstruction(-5).group == "Push Operations" and \
                    block.getInstruction(-4).group == "Duplication Operations" and \
                    block.getInstruction(-3).name == "EQ" and \
                    block.getInstruction(-2).group == "Push Operations" and\
                    block.getInstruction(-1).name == "JUMPI":
            return True
        return False
        
