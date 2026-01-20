from parse.Contract import Contract
from parse.CfgBuilder import CfgBuilder
from parse.CfgAnalysis import CfgAnalysis
from parse.CfgIdentify import CfgIdentify
from parse.entity import StringCleaner, Logger, Stack
from pyevmasm import disassemble_all
import pandas as pd
import os,time,json,re,csv
import numpy as np

dirpath = os.path.dirname(__file__)

cfgpath = "../dataset/blocks/"
listpath = "../dataset/lists/"



def extract(blockjson):
    blockjson=json.loads(blockjson)
    contract_list = []
    blocks = blockjson["basicblocks"]
    removedig = re.compile(r'[0-9]+')
    for block in blocks:
        if "retain" not in block or block["retain"] == True:
            ins = block["instructions"]
            blocklist = []
            for i in ins:
                instr = i["opname"]
                word = removedig.sub('',instr)
                if word != "":
                    blocklist.append(word)
            if blocklist != []:
                contract_list.append(blocklist)
    return contract_list

def extract_oplist(input,output):
    start = time.time()
    dataset = pd.read_csv(cfgpath + input,index_col="address")
    count = [0,0,6166]
    faillist = []

    with open(listpath + output,"w")as fp:
        fieldnames = ['address','oplist']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for i in dataset.index:
            try:
                blockjson = dataset.loc[i,"basicblocks"]
                list = extract(blockjson)
                x = {"address":i,
                        "oplist":json.dumps(list)}
                writer.writerow(x)
            except Exception as e:
                print(e)
                faillist.append(i)
                count[1] += 1
            else:
                count[0] += 1
            finally:
                print(count)
    end = time.time()
    use = end-start
    print(use)

def extact_test():
    dataset = pd.read_csv(cfgpath + "identifiedcfgs.csv",index_col="address")
    builder = CfgBuilder()
    i = "0x1a19c2aec934eb39c92cff0f1ba46efe8f6c56fe"
    blockjson = dataset.loc[i,"basicblocks"]
    cfg = builder.rebuildCfg(blockjson)
    # cfg.storetxt()
    analysis = CfgAnalysis(cfg)
    analysis.analyse()
    l=analysis.extract_list()
    print(l)


if __name__ == "__main__":
    # rawcfgs removedcfgs identifiedcfgs filteredcfgs
    # rawcfglists removedcfglists identifiedcfglists filteredcfglists
    extract_oplist("rawcfgs.csv","rawcfglists.csv")
    extract_oplist("removedcfgs.csv","removedcfglists.csv")
    extract_oplist("identifiedcfgs.csv","identifiedcfglists.csv")
    extract_oplist("filteredcfgs.csv","filteredcfglists.csv")
    extract_oplist("icwscfgs.csv","icwslists.csv")