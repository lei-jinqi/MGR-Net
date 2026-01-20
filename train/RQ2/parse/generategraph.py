import networkx as nx
import pandas as pd
import os,json,math,csv

THRESHOLD = 50

blockpath = "C:/Users/Administrator/Desktop/黄琮雄-科研资资料/dataset/blocks/filteredcfgs.csv"
listspath = "C:/Users/Administrator/Desktop/黄琮雄-科研资资料/dataset/graphlists/"

# 这个函数用于由基本块生成有向图
#这个函数从基本块（basicblocks）中生成一个有向图。每个基本块的偏移量 offset 被映射为图中的节点，并且根据基本块的后继 successors 创建图中的边。这个函数生成一个基础的控制流图。
def generatesimpleGraph(basicblocks):
    g = nx.DiGraph()
    maps = {}
    i = 0
    
    for block in basicblocks:
        maps[block["offset"]] = i
        g.add_node(i)
        i += 1
    for block in basicblocks:
        for suc in block["successors"]:
            # print(type(block["offset"]))
            g.add_edge(maps[block["offset"]],maps[suc])
    return g

# 生成过滤后的图.这个函数基于 generatesimpleGraph 的逻辑，生成过滤后的有向图。它会将某些不需要的节点（通过 retain 标志判断）移除，并通过 removenode 函数调整图的结构，重新连接删除节点的前驱和后继节点。
def generatefilteredGraph(basicblocks):
    g = nx.DiGraph()
    maps = {}
    i = 0
    removednode = []
    
    for block in basicblocks:
        maps[block["offset"]] = i
        g.add_node(i)
        i += 1
        if block["retain"] == False:
            removednode.append(maps[block["offset"]])
    for block in basicblocks:
        for suc in block["successors"]:
            # print(type(block["offset"]))
            g.add_edge(maps[block["offset"]],maps[suc])
    for node in removednode:
        removenode(g,node)
    return g

# 移除节点
def removenode(h,node):
    if node in h:
        innode = list(h.predecessors(node))
        outnode = list(h.successors(node))
        for i in innode:
            for j in outnode:
                if i != j:
                    h.add_edge(i,j)
        h.remove_node(node)

# 这个函数是合并节点中的递归方法.这是一个递归函数，检查当前节点是否可以与前驱节点合并。如果满足条件（即当前节点的入度和出度均为1，并且其后继节点的入度也为1），则会递归合并节点。否则，它将继续处理当前节点及其后继节点
def deep(G,H,pre,now,visited):
    if now in visited:
        H.add_edge(pre,now)
        return        
    else:
        visited.append(now)
    if G.in_degree(now) == 1 and G.out_degree(now) == 1 and G.in_degree(list(G.successors(now))[0]) == 1:
        deep(G,H,pre,list(G.successors(now))[0],visited)
    else:
        if pre is None:
            H.add_node(now)
        else:
            H.add_edge(pre,now)
        if G.out_degree(now) != 0:
            for i in G.successors(now):
                deep(G,H,now,i,visited)

# 需要重新标记.这个函数为图中的每个节点分配一个新的标号。它会创建一个新的图 h，并通过映射表 maps 为原图中的每个节点生成新的索引，然后按照新的索引重新构建图中的边。
def relabelGraph(G):
    h = nx.DiGraph()
    maps=dict()
    i=0

    for node in G.nodes():
        maps[node] = i
        h.add_node(i)
        i += 1
    for node in G.nodes():
        for suc in G.successors(node):
            h.add_edge(maps[node],maps[suc])
    return h

# 合并节点函数
def merge_nodes(G):
    H = nx.DiGraph()
    node =list(G.nodes)[0]
    deep(G,H,None,node,[])
    return H

def getGraph(basicblocks):
    # g = generatesimpleGraph(basicblocks)
    g= generatefilteredGraph(basicblocks)
    # if len(g) > THRESHOLD:
    #     g = merge_nodes(g)
    r = relabelGraph(g)
    return r


def extract_cfglist(file):
    data = pd.read_csv(blockpath,index_col="address")
    print("load data finished")

    success = 0
    total = len(data.index)
    fail = 0
    faillist=[]
    sh=[]

    cols = ['address','edges']
    df = pd.DataFrame(columns=cols)

    for ind in data.index:
        try:
            blockjson = json.loads(data.loc[ind,"basicblocks"])
            g = getGraph(blockjson["basicblocks"])
            if len(g) ==0:
                sh.append(ind)
            newline = {"address":ind,"nodes":json.dumps(list(g.nodes)),"edges":json.dumps(list(g.edges))}
            newrow = pd.Series(newline)
            df=df.append(newrow,ignore_index=True)

            success += 1

        except Exception as e:
            print(ind,e)
            faillist.append(ind)
            fail += 1
        finally:
            print(success, " / ", total)
            print(fail)
    df.to_csv(listspath+file,index=False)
    print(faillist)
    print(sh)


if __name__ == "__main__":
     extract_cfglist("filteredcfg_nomerge_graph.csv")
    # extract_cfglist("filteredcfg_merge_graph.csv")

    
    