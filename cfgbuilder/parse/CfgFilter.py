from parse.Cfg import Cfg
from parse.BlockType import BlockType
from parse.BasicBlock import BasicBlock
from parse.entity import OrderDict,Triplet,Stack,Logger
import re

# 对cfg中的dispatcher、fallback以及函数入口进行查找
# 最新的在analysis中
class CfgFilter:
    def __init__(self):
        pass

    def filter(self,cfg:Cfg):
        if cfg.short:
            return
        self.filter_dispatcher(cfg)
        
    def filter_dispatcher(self,cfg:Cfg):
        if cfg.dispatcher_num > 0 and cfg.dispatcher_num / cfg.length < 0.5:
            for o in cfg.dispatchers:
                block:BasicBlock = cfg.getblock(o)
                block.setretain(False)

    def filter_fallback(self,cfg:Cfg):
        if cfg.fallback_num > 0 and cfg.fallback_num / cfg.length < 0.5:
            for o in cfg.functions:
                block:BasicBlock = cfg.getblock(o)
                if block.has_caller():
                    pass
                block.setretain(False)

    def filter_loader(self,cfg:Cfg):
        if len(cfg.loaders) :
            for o in cfg.loaders:
                block:BasicBlock = cfg.getblock(o)
                block.setretain(False)

    def filter_function(self,cfg:Cfg):
        pass