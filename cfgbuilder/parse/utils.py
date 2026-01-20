import json
from pyevmasm import disassemble_all

def disassemble(bytecode):
    if isinstance(bytecode, str):
        bytecode = bytecode.replace('\n', '')
        if bytecode.startswith('0x'):
            bytecode = bytes.fromhex(bytecode[2:])
        else:
            bytecode = bytes.fromhex(bytecode)
    return disassemble_all(bytecode)

def jsontodot(data,filename):
    nodes = data["runtimeCfg"]["nodes"]
    successors = data["runtimeCfg"]["successors"]
    with open('{}{}.dot'.format(filename, 'json'), 'w') as f:
        f.write('digraph{\n')
        for node in nodes:
            f.write('{}[label="{}"]\n'.format(node["offset"], node["parsedOpcodes"]))
        
        for successor in successors:
            edgefrom = successor["from"]
            edgetos = successor["to"]
            tos = set(edgetos)
            for to in tos:
                f.write('{} -> {}\n'.format(edgefrom, to))  
        f.write('\n}')
