from typing import Tuple
from random import shuffle
import numpy as np
from common.utils import Logger, Config
from common.data_reader.mongodb import MongoDBConnect


def load_ids(
        col_details: Tuple[str, str, str],
        data_split: Tuple = (60, 40),
        sort_by: dict = None,
        limit: int = None,
        shuffle_data: bool = False,
        shuffle_steps: int = 1,
        mongodb_filter: dict = {}):
    """
    Load MongoDB Document Ids from a collection and split them in training and validation data set
    :param col_details: MongoDB collection details with a tuple of 3 string entries
                        [client name (from config), database name, collection name]
    :param data_split: Tuple of percentage of training and test data e.g. (60, 40) for 60% training and 40% test data
    :param sort_by: MongoDB sort expression. e.g. { created_at: -1 }
    :param limit: maximum number of ids that should be fetched
    :param shuffle_data: determine if dataset should be shuffled before splitting it to train and validation data
    :param shuffle_steps: step size for the shuffling (e.g. for time series you want to have a shuffle_size of
                          BATCH_SIZE + (TIME_STEPS - 1)
    :param mongodb_filter: apply to search when finding all ids
    :return: training and validation data
    """
    Logger.logger.info(f"Loading Document IDs from MongoDB: {col_details[2]}")
    mongo_con = MongoDBConnect()
    mongo_con.add_connections_from_config(Config.get_config_parser())
    collection = mongo_con.get_collection(*col_details)

    if sort_by is None:
        sort_by = {"_id": 1}

    db_cursor = collection.find(mongodb_filter, sort_by)

    if limit:
        db_cursor.limit(limit)
    tmp_docs = []
    for doc in db_cursor:
        tmp_docs.append(doc["_id"])

    if shuffle_data:
        if shuffle_steps == 1:
            shuffle(tmp_docs)
        else:
            # if reshape the tmp_docs must be a multiple of shuffle_steps, cut ids that do no fit
            overflow = len(tmp_docs) % shuffle_steps
            tmp_docs = tmp_docs[:len(tmp_docs) - overflow]
            x = np.reshape(tmp_docs, (-1, shuffle_steps))
            np.random.shuffle(x)
            tmp_docs = x.flatten().tolist()

    train_range = int((data_split[0] / 100) * len(tmp_docs))
    train_data = tmp_docs[:train_range]
    val_data = tmp_docs[train_range:]
    Logger.logger.info("Documents loaded (train|validation): {0} | {1}\n\n".format(
        len(train_data), len(val_data)))

    return train_data, val_data
