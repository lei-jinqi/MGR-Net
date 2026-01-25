from parse.Contract import Contract
from parse.CfgBuilder import CfgBuilder
from parse.CfgAnalysis import CfgAnalysis
from parse.CfgIdentify import CfgIdentify
from parse.entity import StringCleaner, Logger, Stack
from pyevmasm import disassemble_all
import pandas as pd
import os, time, json
import csv
import numpy as np
import gensim

dirpath = os.path.dirname(__file__)

datapath = "../dataset/dataset_6492.csv"
blockpath = "../dataset/blocks/"

TOTAL_NUM = 6492


def generaterawcfg():
    start = time.time()
    dataset = pd.read_csv(datapath, index_col="address")
    count = [0, 0, TOTAL_NUM]
    faillist = []

    with open(blockpath + "rawcfgs.csv", "w") as fp:
        fieldnames = ['address', 'basicblocks']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        builder = CfgBuilder()
        for i in dataset.index:
            bytecode = dataset.loc[i, "bytecode"]
            bytecode = StringCleaner.removeInfo(bytecode)
            try:
                cfg = builder.buildCfg(i, bytecode)
                blockjson = cfg.storejson()

                x = {"address": i,
                     "basicblocks": json.dumps(blockjson)}

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
    use = end - start
    with open(dirpath + "/faillist.txt", "w") as f:
        for i in faillist:
            f.write(i + "\n")


# 生成过滤孤儿块的控制流图
# removedcfgs.csv
def generateremovedcfg():
    start = time.time()
    dataset = pd.read_csv(datapath, index_col="address")
    count = [0, 0, TOTAL_NUM]
    faillist = []

    with open(blockpath + "removedcfgs.csv", "w") as fp:
        fieldnames = ['address', 'basicblocks']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        builder = CfgBuilder()
        for i in dataset.index:
            bytecode = dataset.loc[i, "bytecode"]
            bytecode = StringCleaner.removeInfo(bytecode)
            try:
                cfg = builder.buildCfg(i, bytecode)
                blockjson = cfg.storejson()

                x = {"address": i,
                     "basicblocks": json.dumps(blockjson)}

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
    use = end - start
    with open(dirpath + "/faillist.txt", "w") as f:
        for i in faillist:
            f.write(i + "\n")


# 生成识别后的dispatcher，fallback，以及各个函数的控制流图
# 这个控制流图会用于后续的analysis分析，过滤，以及特征提取
# identifiedcfg
def generateidentifiedcfgs():
    start = time.time()
    dataset = pd.read_csv(datapath, index_col="address")
    count = [0, 0, TOTAL_NUM]
    faillist = []

    with open(blockpath + "identifiedcfgs.csv", "w") as fp:
        fieldnames = ['address', 'basicblocks']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        builder = CfgBuilder()
        identify = CfgIdentify()
        for i in dataset.index:

            try:
                bytecode = dataset.loc[i, "bytecode"]

                if not isinstance(bytecode, str):
                    raise ValueError(f"Bytecode for {i} is not a string (is {type(bytecode)}). Skipping.")

                bytecode = StringCleaner.removeInfo(bytecode)

                cfg = builder.buildCfg(i, bytecode)
                cfg = identify.identify(cfg)
                blockjson = cfg.storejson()

                x = {"address": i,
                     "basicblocks": json.dumps(blockjson)}

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
    use = end - start
    with open(dirpath + "/faillist.txt", "w") as f:
        for i in faillist:
            f.write(i + "\n")


def generatefiltercfgs():
    start = time.time()
    dataset = pd.read_csv(blockpath + "identifiedcfgs.csv", index_col="address")

    count = [0, 0, TOTAL_NUM]
    faillist = []
    with open(blockpath + "filteredcfgs3.csv", "w") as fp:
        fieldnames = ['address', 'basicblocks']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        builder = CfgBuilder()
        for i in dataset.index:
            try:
                blockjson = dataset.loc[i, "basicblocks"]
                cfg = builder.rebuildCfg(blockjson)
                analysis = CfgAnalysis(cfg)

                analysis.analyse()

                blockjson = analysis.cfg.storejson()
                x = {"address": i,
                     "basicblocks": json.dumps(blockjson)}

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
    use = end - start
    with open(dirpath + "/faillist.txt", "w") as f:
        for i in faillist:
            f.write(i + "\n")


def generateicws():
    start = time.time()
    dataset = pd.read_csv(blockpath + "identifiedcfgs.csv", index_col="address")

    count = [0, 0, TOTAL_NUM]
    faillist = []
    with open(blockpath + "icwscfgs.csv", "w") as fp:
        fieldnames = ['address', 'basicblocks']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        builder = CfgBuilder()
        for i in dataset.index:
            try:
                blockjson = dataset.loc[i, "basicblocks"]
                cfg = builder.rebuildCfg(blockjson)
                analysis = CfgAnalysis(cfg)

                analysis.analyse_icws()

                blockjson = analysis.cfg.storejson()
                x = {"address": i,
                     "basicblocks": json.dumps(blockjson)}

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
    use = end - start
    with open(dirpath + "/faillist.txt", "w") as f:
        for i in faillist:
            f.write(i + "\n")


# 主函数
if __name__ == "__main__":
    generaterawcfg()
    generateremovedcfg()
    generateidentifiedcfgs()
    generatefiltercfgs()
    generateicws()
