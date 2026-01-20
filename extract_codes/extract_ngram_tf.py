import pandas as pd
import os,json,math,csv

listpath = "../dataset/lists/"
midpath = "../dataset/features/mid_product/"
featurepath = midpath
TOTAL_NUM = 6166

def extract_tf(tcfile,ngramfile,storefile):

    data = pd.read_csv(midpath+tcfile, index_col="address")
    print("load data",tcfile)
    cols = getngramlist(ngramfile)
    feature_cols = cols["1gram"]+cols["2gram"]+cols["3gram"]+cols["4gram"] +\
     ["1gram_counts","2gram_counts","3gram_counts","4gram_counts"]

    with open(featurepath + storefile,"w")as fp:
        fieldnames = ['address'] + feature_cols
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        success = 0
        total = len(data.index)
        fail = 0
        faillist=[]

        for ind in data.index:
            try:
                x = {"address":ind}
                # 1gram
                sum = data.loc[ind,"1gram_counts"]
                for op in cols["1gram"]:
                    c=data.loc[ind,op]
                    if c == 0 or sum == 0:
                        x[op] = 0.0
                    else:
                        x[op] = (c*1.0)/(sum*1.0)
                x["1gram_counts"] = sum
                # 2gram
                sum = data.loc[ind,"2gram_counts"]
                for op in cols["2gram"]:
                    c=data.loc[ind,op]
                    if c == 0 or sum == 0:
                        x[op] = 0.0
                    else:
                        x[op] = (c*1.0)/(sum*1.0)
                x["2gram_counts"]=sum
                # 3gram
                sum = data.loc[ind,"3gram_counts"]
                for op in cols["3gram"]:
                    c=data.loc[ind,op]
                    if c == 0 or sum == 0:
                        x[op] = 0.0
                    else:
                        x[op] = (c*1.0)/(sum*1.0)
                x["3gram_counts"]=sum
                # 4gram
                sum = data.loc[ind,"4gram_counts"]
                for op in cols["4gram"]:
                    c=data.loc[ind,op]
                    if c == 0 or sum == 0:
                        x[op] = 0.0
                    else:
                        x[op] = (c*1.0)/(sum*1.0)
                x["4gram_counts"]=sum

                writer.writerow(x)
                success +=1
            except Exception as e:
                fail+=1
                faillist.append(ind)
            finally:
                print(success," / ",fail, " / ", total)
        print(faillist)


def getngramlist(name):
    path = midpath + name
    with open(path)as fp:
        j=json.load(fp)
    return j




if __name__ == "__main__":
    # extract_tf("rawcfgs_ngram_count.csv","rawcfg_ngram_term_count.json","rawcfg_ngram_tf.csv")
    # extract_tf("removedcfgs_ngram_count.csv","removedcfg_ngram_term_count.json","removedcfg_ngram_tf.csv")
    extract_tf("filteredcfgs_ngram_count.csv","filteredcfg_ngram_term_count.json","filteredcfg_ngram_tf.csv")

