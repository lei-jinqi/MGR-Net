import re, logging

path = "./logs/"


class Logger:
    __slots__ = ['logger', 'contractlist']

    def __init__(self, name):
        self.contractlist = set()
        filename = path + name + ".log"
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.ERROR)
        if not self.logger.handlers:
            filehandler = logging.FileHandler(filename, "w")
            streamhandler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            filehandler.setFormatter(formatter)
            streamhandler.setFormatter(formatter)
            self.logger.addHandler(streamhandler)
            self.logger.addHandler(filehandler)

    def addDirectJumpTargetErrors(self, cn, source, target):
        self.logger.info("{} DirectJumpTargetErrors at {:x},{:x} is None".format(cn, source, target))

    def addOrphanJumpTargetNullErrors(self, cn, source, target):
        self.contractlist.add(cn)
        self.logger.error("{} OrphanJumpTargetNullErrors at {:x},{:x} is None".format(cn, source, target))

    def addOrphanJumpTargetUnknownErrors(self, cn, source):
        self.contractlist.add(cn)
        self.logger.error(
            "{} OrphanJumpTargetUnknownErrors at {:x},symbolic execution found an unknown value".format(cn, source))

    def addLoopDepthExceededErrors(self, cn, source):
        self.contractlist.add(cn)
        self.logger.error("{} LoopDepthExceededErrors at {:x}".format(cn, source))

    def addBlockLimitErrors(self, cn, limit):
        self.contractlist.add(cn)
        self.logger.error("{} BlockLimitErrors : {}".format(cn, limit))

    def addMultipleRootNodesErrors(self, cn):
        self.contractlist.add(cn)
        self.logger.error("{} MultipleRootNodesErrors".format(cn))

    def addStackExceededErrors(self, cn, block, offset):
        self.contractlist.add(cn)
        self.logger.error("{} StackExceededErrors at {:x},offset:{:x}".format(cn, block, offset))

    def addCriticalErrors(self, cn):
        self.contractlist.add(cn)
        self.logger.error("{} CriticalErrors".format(cn))

    def addBuildTimeMillis(self, cn):
        self.contractlist.add(cn)
        self.logger.error("{} BuildTimeMillis".format(cn))


class Pair:

    def __init__(self, key, value):
        self._key = key
        self._value = value

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def __repr__(self):
        return "<{}:{}>".format(self.key, self.value)

    def __str__(self) -> str:
        return "<{}:{}>".format(self.key, self.value)

    def __iter__(self):
        yield self._key
        yield self._value


class OrderDict:

    def __init__(self):
        self.orderSet = []

    @property
    def length(self):
        return len(self.orderSet)

    @property
    def start(self) -> Pair:
        if self.length > 0:
            return self.orderSet[0]

    @property
    def end(self) -> Pair:
        if self.length > 0:
            return self.orderSet[-1]

    def update(self, key, value):
        for i in range(self.length):
            if self.orderSet[i].key == key:
                self.orderSet[i].setValue(value)
                return
        self.orderSet.append(Pair(key, value))

    def remove(self, key):
        for i in range(self.length):
            if self.orderSet[i].key == key:
                del self.orderSet[i]
                return

    def get(self, key):
        for tp in self.orderSet:
            if tp.key == key:
                return tp.value

    def getbyindex(self, index):
        if index < self.length and index > self.length * -1:
            return self.orderSet[index]

    def keys(self):
        for tp in self.orderSet:
            yield tp.key

    def __setitem__(self, key, value):
        for i in range(self.length):
            if self.orderSet[i].key == key:
                self.orderSet[i].setValue(value)
                return
        self.orderSet.append(Pair(key, value))

    def __getitem__(self, key):
        for tp in self.orderSet:
            if tp.key == key:
                return tp.value

    def __delitem__(self, key):
        for i in range(len(self.orderSet)):
            if self.orderSet[i].key == key:
                del self.orderSet[i]
                return

    def __repr__(self):
        return repr(self.orderSet)

    def __str__(self) -> str:
        return str(self.orderSet)

    def __iter__(self):
        for tp in self.orderSet:
            yield tp


class Triplet:
    __slot__ = ['_elem1', '_elem2', '_elem3']

    def __init__(self, e1, e2, e3):
        self._elem1 = e1
        self._elem2 = e2
        self._elem3 = e3

    @property
    def elem1(self):
        return self._elem1

    @property
    def elem2(self):
        return self._elem2

    @property
    def elem3(self):
        return self._elem3

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return "<{},{},{}>".format(self._elem1, self._elem2, self._elem3)

    def __hash__(self):
        return hash(str(self.elem1) + str(self.elem2) + str(self.elem3))

    def __eq__(self, other) -> bool:
        if isinstance(other, Triplet):
            if id(self) == id(other): return True
            if other == None: return False
            return self.elem1 == other.elem1 and \
                self.elem2 == other.elem2 and \
                self.elem3 == other.elem3
        return False


class Stack:

    def __init__(self):
        self._list = []

    def isEmpty(self) -> bool:
        return self._list == []

    def push(self, item):
        self._list.append(item)

    def pop(self):
        return self._list.pop()

    def peek(self):
        if self._list:
            return self._list[-1]
        else:
            return None

    def size(self):
        return len(self._list)

    def __str__(self) -> str:
        return str(self._list)

    def __repr__(self) -> str:
        return repr(self._list)

    def __eq__(self, other) -> bool:
        if isinstance(other, Stack):
            if id(self) == id(other): return True
            if other is None: return False
            return self._list == other._list
        return False


class Instruct:

    def __init__(self, pc, opname, operand):
        self.pc = pc
        self.name = opname
        self.operand = operand

    def __str__(self) -> str:
        if self.operand is not None:
            return "{} {}".format(self.name, hex(self.operand))
        else:
            return "{}".format(self.name)


class StringCleaner:
    state = re.compile(r'^(73[0-9a-fA-F]{40}3014)?60(60|80)604052[0-9a-fA-F]*$')
    deploy = re.compile(r'(0396000f3|0396000f300|0396000f3fe)(?=60(60|80)604052)')
    auxdata1 = re.compile(r'a165627a7a72305820[0-9a-f]{64}0029')
    auxdata2 = re.compile(r'a265627a7a72(30|31)5820[0-9a-f]{64}64736f6c6343[0-9a-f]{6}0032"')
    auxdata3 = re.compile(r'a264697066735822[0-9a-f]{68}64736f6c6343[0-9a-f]{6}00(32|33)')

    def __init__(self):
        pass

    @staticmethod
    def matchstate(source):
        if StringCleaner.state.match(source):
            return True
        else:
            return False

    @staticmethod
    def matchdeploy(source):
        if StringCleaner.deploy.search(source):
            return True
        else:
            return False

    @staticmethod
    def removedeploy(source):
        if StringCleaner.matchdeploy(source):
            return StringCleaner.deploy.split(source, 1)[3]
        else:
            return source

    @staticmethod
    def removeauxdata(source):
        if StringCleaner.auxdata1.search(source):
            return StringCleaner.auxdata1.split(source, 1)[0]
        elif StringCleaner.auxdata2.search(source):
            return StringCleaner.auxdata2.split(source, 1)[0]
        elif StringCleaner.auxdata3.search(source):
            return StringCleaner.auxdata3.split(source, 1)[0]
        else:
            return source

    @staticmethod
    def removeInfo(source):
        temp = StringCleaner.removedeploy(source)
        return StringCleaner.removeauxdata(temp)
