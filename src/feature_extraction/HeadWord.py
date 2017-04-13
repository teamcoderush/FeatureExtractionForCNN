from nltk.parse.stanford import StanfordDependencyParser
import pandas as pd

path_to_jar = '../../lib_external/stanford_parser/stanford-parser.jar'
path_to_models_jar = '../../lib_external/stanford_parser/stanford-parser-3.7.0-models.jar'

# Load dataset
url = "../../data/csv/ABSA16_Restaurants_Train_SB1_v2.csv"  # relative dataset URL
dataset = pd.read_csv(url, encoding='latin1')  # reads dataset with headers
train = dataset.groupby('text', as_index=False)['category'].agg({'categories': (lambda x: list(x))})

dependancy_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

_out = []

for i in train.text:
    result = dependancy_parser.raw_parse(i)
    for parse in result:
        _out.append(parse.tree()._label)
        #print(parse.tree()._label)

