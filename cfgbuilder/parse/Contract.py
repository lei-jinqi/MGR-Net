
from parse.CfgBuilder import CfgBuilder
from parse.entity import StringCleaner

class Contract:
    
    __slots__ =['_name','_bytecode','_Cfg']

    def __init__(self,name:str,bytecode) -> None:
        self._name =name
        self._bytecode = StringCleaner.removeInfo(bytecode)
        self._Cfg = CfgBuilder.buildCfg(self._name,self._bytecode)
        

    @property
    def name(self):
        return self._name
    
    @property 
    def bytecode(self):
        return self._bytecode

    @property
    def cfg(self):
        return self._Cfg
    
    @property
    def raw(self):
        return self._Cfg.raw
    