import pandas as pd
import os, json, math, csv

listpath = "../dataset/lists/"
midpath = "../dataset/features/mid_product/"
featurepath = midpath
TOTAL_NUM = 6166


def extract_tfidf(tffile, ngramfile, storefile):
    data = pd.read_csv(featurepath + tffile, index_col="address")
    cols = getngramlist(ngramfile)
    feature_cols = cols["1gram"] + cols["2gram"] + cols["3gram"] + cols["4gram"]

    ni = {}
    try:
        tot = len(feature_cols)
        cc = 0
        for col in feature_cols:
            ni[col] = 0.0
            for ind in data.index:
                if data[col].get(ind) > 0.0:
                    ni[col] += 1.0
            cc += 1
            print("ni", cc, "/", tot)
        print("ni end")
    except Exception as e:
        print("ni error")

    idf = {}
    try:
        tot = len(feature_cols)
        cc = 0
        for col in feature_cols:
            idf[col] = math.log((TOTAL_NUM + 1.0) / (float(ni[col]) + 1.0)) + 1.0
            cc += 1
            print("idf:", cc, "/", tot)
        print("idf end")
    except Exception as e:
        print("idf error")

    tfidf = []
    succ = 0
    fail = 0
    tot = TOTAL_NUM
    for ind in data.index:
        x = {"address": ind}
        try:
            for col in feature_cols:
                newcol = col + "_tfidf"
                x[newcol] = float(data[col].get(ind)) * idf[col]
            tfidf.append(x)
            succ += 1
        except Exception as e:
            print(e)
            fail += 1
        finally:
            print("tfidf:", succ, fail, tot)
    print("store start")
    data = pd.DataFrame(tfidf)
    data.to_csv(featurepath + storefile, index=False)
    print("store end")


def getngramlist(name):
    path = midpath + name
    with open(path) as fp:
        j = json.load(fp)
    return j


if __name__ == "__main__":
    # extract_tfidf("rawcfg_ngram_tf.csv", "rawcfg_ngram_term_count.json", "rawcfg_ngram_tfidf.csv")
    # extract_tfidf("removedcfg_ngram_tf.csv", "removedcfg_ngram_term_count.json", "removedcfg_ngram_tfidf.csv")
    extract_tfidf("filteredcfg_ngram_tf.csv", "filteredcfg_ngram_term_count.json", "filteredcfg_ngram_tfidf.csv")
