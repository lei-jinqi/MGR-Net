from pyevmasm.evmasm import Instruction

class SymbolicStack:
    MAX_STACK_SIZE:int = 1024
    STACK_TAIL_SIZE:int = 48
    STACK_TAIL_THRESHOLD:int = 200

    def __init__(self):
        self._stack = []

    def execuate(self,op:Instruction ):
        if self.length > self.MAX_STACK_SIZE:
            raise Exception("Exceeded")
        if op.group == "Push Operations":
            self.execuatePush(op)
        elif op.group == "Duplication Operations":
            self.execuateDup(op)
        elif op.group == "Exchange Operations":
            self.execuateSwap(op)
        elif op.opcode == 0x50:
            self.execuatePop()
        elif op.opcode == 0x16:
            self.execuateAnd(op)
        else:
            if self.length <= op.pops:
                self._stack.clear()
            else:
                for i in range(op.pops):
                    self._stack.pop()
            for i in range(op.pushes):
                self._stack.append(None)

    def execuatePush(self,op:Instruction):
        self._stack.append(op.operand)

    def execuateDup(self,op:Instruction):
        index = self.length - op.pops
        if index >= 0:
            x = self._stack[self.length - op.pops]
            self._stack.append(x)
        # else:
        #     self._stack.append(None)

    def execuateSwap(self,op:Instruction):
        i = self.length - op.pops
        j = self.length - 1
        if i > 0 and j > 0:
            tmp = self._stack[i]
            self._stack[i] = self._stack[j]
            self._stack[j] = tmp
        # else:
        #     self._stack.pop()
        #     self._stack.append(None)

    def execuatePop(self):
        if self.length:
            self.pop()

    def execuateAnd(self,op:Instruction):
        a = self.pop()
        b = self.pop()
        if a == None or b == None:
            self._stack.append(None)
        else:
            self._stack.append(a&b)

    def execuateAdd(self,op:Instruction):
        a = self.pop()
        b = self.pop()
        if a == None or b == None:
            self._stack.append(None)
        else:
            self._stack.append(a+b)
 
    def __repr__(self) -> str:
        return repr(self._stack)

    def __str__(self) -> str:
        list_str = "["
        for i in self._stack:
            if i is not None:
                list_str += "0x{:x},".format(i)
            else:
                list_str += "None,"
        if len(list_str) > 1:
            list_str = list_str[:-1]
        list_str += "]"
        return list_str

    def getStack(self):
        return self._stack

    @property
    def length(self):
        return len(self._stack)
    
    def clear(self):
        self._stack = []

    def copy(self):
        result = SymbolicStack()
        result.getStack().extend(self._stack)
        return result

    def peek(self):
        if self.length:
            return self._stack[-1]

    def pop(self):
        if len(self._stack):
            value = self._stack.pop()
            return value

    def __eq__(self, other) -> bool:
        if isinstance(other,SymbolicStack):
            if id(self) == id(other): return True
            that:SymbolicStack = other
            stackTailSize = self.STACK_TAIL_SIZE / ((self.length - self.STACK_TAIL_THRESHOLD) / 100)  if self.length >= self.STACK_TAIL_THRESHOLD + 100 else self.STACK_TAIL_SIZE
            stackTailSize = int(stackTailSize)
            if self.length < stackTailSize and that.length < stackTailSize:
                return self._stack == that.getStack()
            else:
                return self._stack[self.length - stackTailSize:self.length] == that.getStack()[that.length-stackTailSize:that.length]

        return False

def test():
    s1 = SymbolicStack()
    s2 = SymbolicStack()
    a = Instruction(0x60,"PUSH",1,0,1,3,"",0x0)
    b = Instruction(0x63,"PUSH",4,0,1,3,"",0x70a08231)
    c1 = Instruction(0x60,"PUSH",1,0,1,3,"",0x40)
    c2 = Instruction(0x60,"PUSH",1,0,1,3,"",0x40)
    d = Instruction(0x60,"PUSH",1,0,1,3,"",0x22)
    e = Instruction(0x60,"PUSH",1,0,1,3,"",0xa0)
    pop = Instruction(0x50,"POP",0,1,0,3,"")
    an = Instruction(0x16,"AND",0,2,1,3,"")
    dup = Instruction(0x81,"DUP",0,1,2,3,"")
    swap = Instruction(0x90,"SWAP",0,5,5,3,"")
    s1.execuate(a)
    s1.execuate(b)
    s1.execuate(c1)
    for i in range(0,500):
        s1.execuate(d)
    s2.execuate(a)
    s2.execuate(b)
    s2.execuate(c2)
    for i in range(0,500):
        s2.execuate(d)
    print(s1)
    print(s2)
    print(s1==s2)

if __name__ == "__main__":
    test()
    


    