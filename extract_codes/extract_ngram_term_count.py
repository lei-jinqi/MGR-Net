import pandas as pd
import os, json, math, csv

listpath = "../dataset/lists/"
midpath = "../dataset/features/mid_product/"


def storengramlist(input, output):
    # rawcfglists removedcfglists filteredcfglists
    path = listpath + input
    data = pd.read_csv(path, index_col="address")
    success = 0
    total = len(data.index)
    fail = 0
    op1 = set()
    op2 = set()
    op3 = set()
    op4 = set()
    for ind in data.index:
        try:
            listjson = data.loc[ind, "oplist"]
            lis = json.loads(listjson)
            for ins in lis:
                length = len(ins)
                for i in ins:
                    op1.add(i)
                if length >= 2:
                    for j in range(0, length - 1):
                        name = ins[j] + "_" + ins[j + 1]
                        op2.add(name)
                if length >= 3:
                    for j in range(0, length - 2):
                        name = ins[j] + "_" + ins[j + 1] + "_" + ins[j + 2]
                        op3.add(name)
                if length >= 4:
                    for j in range(0, length - 3):
                        name = ins[j] + "_" + ins[j + 1] + "_" + ins[j + 2] + "_" + ins[j + 3]
                        op4.add(name)
            success += 1
        except Exception as e:
            print(e)
            fail += 1
        finally:
            print(success, " / ", total)
            print(fail)
    op1gram = list(op1)
    op2gram = list(op2)
    op3gram = list(op3)
    op4gram = list(op4)
    print(len(op1gram), len(op2gram), len(op3gram), len(op4gram))
    oplist = {"1gram": op1gram,
              "2gram": op2gram,
              "3gram": op3gram,
              "4gram": op4gram}
    # rawcfg_ngram_term_count removedcfg_ngram_term_count filteredcfg_ngram_term_count
    with open(midpath + output, "w") as fp:
        json.dump(oplist, fp)


def getngramlist(name):
    path = midpath + name
    with open(path) as fp:
        j = json.load(fp)
    return j


def extract_ngram_term_count(gramlist, cfglists, output):
    # get the opcodes
    # rawcfg_ngram_term_count,removedcfg_ngram_term_count,filteredcfg_ngram_term_count
    cols = getngramlist(gramlist)
    feature_cols = cols["1gram"] + cols["2gram"] + cols["3gram"] + cols["4gram"] + \
                   ["1gram_counts", "2gram_counts", "3gram_counts", "4gram_counts"]

    # open the list file 
    # rawcfglists removedcfglists filteredcfglists
    path = listpath + cfglists
    data = pd.read_csv(path, index_col="address")
    success = 0
    total = len(data.index)
    fail = 0
    # store the count text
    # rawcfgs_ngram_count,removedcfgs_ngram_count filteredcfgs_ngram_count
    with open(midpath + output, "w") as fp:
        fieldnames = ['address'] + feature_cols
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for ind in data.index:
            try:

                x = {"address": ind}
                count = [0, 0, 0, 0]
                for j in feature_cols:
                    x[j] = 0
                listjson = data.loc[ind, "oplist"]
                lis = json.loads(listjson)
                for ins in lis:
                    length = len(ins)
                    for i in ins:
                        x[i] += 1
                        count[0] += 1
                    if length >= 2:
                        for j in range(0, length - 1):
                            name = ins[j] + "_" + ins[j + 1]
                            x[name] += 1
                            count[1] += 1
                    if length >= 3:
                        for j in range(0, length - 2):
                            name = ins[j] + "_" + ins[j + 1] + "_" + ins[j + 2]
                            x[name] += 1
                            count[2] += 1
                    if length >= 4:
                        for j in range(0, length - 3):
                            name = ins[j] + "_" + ins[j + 1] + "_" + ins[j + 2] + "_" + ins[j + 3]
                            x[name] += 1
                            count[3] += 1
                x["1gram_counts"] = count[0]
                x["2gram_counts"] = count[1]
                x["3gram_counts"] = count[2]
                x["4gram_counts"] = count[3]
                writer.writerow(x)

                success += 1
            except Exception as e:
                print(e)
                fail += 1
            finally:
                print(success, " / ", total)
                print(fail)


def readcsv():
    path = midpath + "rawcfgs_ngram_count.csv"
    data = pd.read_csv(path, index_col="address")
    ind = "0xf41624c6465e57a0dca498ef0b62f07cbaab09ca"
    j = data.loc[ind, "1gram_counts"]
    print(j)


if __name__ == "__main__":
    # storengramlist("rawcfglists.csv","rawcfg_ngram_term_count.json")
    # storengramlist("removedcfglists.csv","removedcfg_ngram_term_count.json")
    storengramlist("filteredcfglists.csv","filteredcfg_ngram_term_count.json")

    # extract_ngram_term_count("rawcfg_ngram_term_count.json","rawcfglists.csv","rawcfgs_ngram_count.csv")
    # extract_ngram_term_count("removedcfg_ngram_term_count.json","removedcfglists.csv","removedcfgs_ngram_count.csv")
    extract_ngram_term_count("filteredcfg_ngram_term_count.json","filteredcfglists.csv","filteredcfgs_ngram_count.csv")
