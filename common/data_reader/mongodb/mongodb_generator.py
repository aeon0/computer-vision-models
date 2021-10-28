""" Data Generator for MongoDB """
import math
import numpy as np
from typing import List, Tuple
from random import shuffle

from common.utils import Config, Logger
from common.data_reader.base_data_generator import BaseDataGenerator
from common.data_reader.mongodb import MongoDBConnect
from common.data_reader.i_cache import ICache


class MongoDBGenerator(BaseDataGenerator):
    def __init__(self,
                 col_details: List[Tuple[str, str, str]],
                 doc_ids: List[List[any]],
                 batch_size: int = 32,
                 processors: List[any] = list(),
                 cache: ICache = None,
                 shuffle_data: bool = True,
                 data_group_size: int = 1,
                 continues_data_selection: bool = True,
                 fix_batch_size: bool = False):
        """
        :param col_details: List of MongoDB collection details with a tuple of 3 string entries
                            [client name (from config), database name, collection name]
        :param doc_ids: List of List of doc ids which are used to get the specific data from the MongoDB
        :param batch_size: number of batch size
        :param processors: List of Data processors
        :param cache: Passing instance of a cache e.g. RedisCache, if it is None, no caching is used.
                      Only possible if redis is locally available (has to be installed)
        :param shuffle_data: bool flag to determine if set should be shuffled after epoch is done
        :param data_group_size: number of steps that should be grouped e.g for time series. The data will still only
                                move forward one time step. E.g. for data_group_size=3:
                                [t-5, t-4, t-3], [t-4, t-3, t-2], [t-3, t-2, -1], etc.
                                data will not be shuffled in case data_group_size > 1
        :param fix_batch_size: if true batch size will always be the same, e.g. if batch_size=64 and there are only 63
                               datasamples left for the final batch, these 63 data points will be ignored. In case the
                               batch size of your model is fixed, set this to True.
        """
        super().__init__(batch_size, processors)
        self.doc_ids = doc_ids
        self.cache = cache
        self.shuffle_data = shuffle_data
        self.data_group_size = max(data_group_size, 1)
        self.continues_data_selection = continues_data_selection
        self.docs_per_batch = self.batch_size + (self.data_group_size - 1)
        self.col_details = col_details
        self.fix_batch_size = fix_batch_size
        self.collections = None
        self.mongo_con = MongoDBConnect()
        self.mongo_con.add_connections_from_config(Config.get_config_parser())
        # in case a step_size is chosen > 1, make sure that len(doc_ids) is a multiple of that
        # otherwise reshape will not be working and throw errors
        if self.data_group_size > 1:
            for i in range(len(self.doc_ids)):
                overflow = len(self.doc_ids[i]) % self.docs_per_batch
                self.doc_ids[i] = self.doc_ids[i][:len(self.doc_ids[i]) - overflow]

    def _fetch_data(self, collection, query_docs: list) -> List[any]:
        """
        Get a set of _ids from the database (in order)
        :param query_docs: A list of _ids
        :return: A pymongo cursor
        """
        # to ensure the order of query_docs, use this method. For more details look at this stackoverflow question:
        # https://stackoverflow.com/questions/22797768/does-mongodbs-in-clause-guarantee-order/22800784#22800784
        # query = [
        #     {"$match": {"_id": {"$in": query_docs}}},
        #     {"$unwind": {"path": '$assigned', "preserveNullAndEmptyArrays": True }},
        #     {"$addFields": {"__order": {"$indexOfArray": [query_docs, "$_id"]}}},
        #     {"$sort": {"__order": 1}}
        # ]
        # docs = collection.aggregate(query)
        docs = []
        for doc_id in query_docs:
            docs.append(collection.find_one({"_id": doc_id}))
        return docs

    def __len__(self) -> int:
        """
        :return: Number of batches per epoch
        """
        num_ids = sum(len(ids) for ids in self.doc_ids)
        if self.fix_batch_size:
            num_batches = int(math.floor(num_ids / self.docs_per_batch))
        else:
            num_batches = int(math.ceil(num_ids / self.docs_per_batch))
        return num_batches

    def __getitem__(self, idx):
        """
        Get batch data
        :param idx: current idx in the batch
        :return: arrays for traning_data (x) and labels (y)
        """

        # Connections should always be established on first __getitem__ class to support multiprocessing
        # every fork needs to have its own database connection
        if self.collections is None:
            self.collections = []
            for detail in self.col_details:
                self.collections.append(self.mongo_con.get_collection(*detail))

        # based on the batch_idx and each doc_ids entry length, check where to take the ids from
        data_start_idx = idx * self.docs_per_batch
        collection = None
        batch_ids = []
        for i in range(len(self.doc_ids)):
            len_docs = len(self.doc_ids[i])
            if len_docs <= data_start_idx:
                data_start_idx -= len_docs
            else:
                collection = self.collections[i]
                batch_ids = self.doc_ids[i][data_start_idx : data_start_idx + self.docs_per_batch]
                break

        if self.cache is not None and self.cache.exist(batch_ids):
            docs = self.cache.get(batch_ids)
        else:
            docs = self._fetch_data(collection, batch_ids)
            if self.cache is not None:
                # save fetched data to cache
                for doc in docs:
                    success = self.cache.set(str(doc["_id"]), doc)
                    if not success:
                        Logger.logger.warning("Redis cache is full")
                        break
                # Create new command cursor since the original one is finished after looping once
                docs = self._fetch_data(collection, batch_ids)

        if self.data_group_size > 1:
            # reshape to fit step_size and copy data
            # since docs is a cursor, save in a temporary list  
            tmp_data = list(docs)
            docs = []
            start_idx = 0
            end_idx = self.data_group_size
            while end_idx <= len(tmp_data):
                if self.continues_data_selection:
                    docs.append(tmp_data[start_idx:end_idx])
                    start_idx += 1
                    end_idx += 1
                else:
                    docs.append(tmp_data[start_idx:end_idx])
                    start_idx += self.data_group_size
                    end_idx += self.data_group_size

        batch_x, batch_y = self._process_batch(docs)
        input_data = []
        if len(batch_x) > 0:
            if isinstance(batch_x[0], dict):
                # multiple inputs, split them up by name
                input_data = {}
                for key in batch_x[0]:
                    input_data[key] = []
                # fill dict with data for each key
                for batch in batch_x:
                    for key in batch:
                        input_data[key].append(np.asarray(batch[key]))
            if isinstance(batch_x[0], list):
                # multiple inputs, split them up by index
                len_inputs = len(batch_x[0])
                for i in range(len_inputs):
                    input_data.append([])
                for batch in batch_x:
                    for i in range(len_inputs):
                        input_data[i].append(np.asarray(batch[i]))
                for i in range(len_inputs):
                    input_data[i] = np.asarray(input_data[i])
            else:
                input_data = np.asarray(batch_x)

        ground_truth = []
        if len(batch_y) > 0:
            if isinstance(batch_y[0], dict):
                # multiple inputs, split them up by name
                ground_truth = {}
                for key in batch_y[0]:
                    ground_truth[key] = []
                # fill dict with data for each key
                for batch in batch_y:
                    for key in batch:
                        ground_truth[key].append(np.asarray(batch[key]))
            if isinstance(batch_y[0], list):
                # multiple inputs, split them up by index
                len_outputs = len(batch_y[0])
                for i in range(len_outputs):
                    ground_truth.append([])
                for batch in batch_y:
                    for i in range(len_outputs):
                        ground_truth[i].append(np.asarray(batch[i]))
                for i in range(len_outputs):
                    ground_truth[i] = np.asarray(ground_truth[i])
            elif batch_y[0] is not None:
                ground_truth = np.asarray(batch_y)

        if len(ground_truth) == 0:
            ground_truth = None

        return (input_data, ground_truth)

    def on_epoch_end(self):
        """
        Called after each epoch
        """
        self.epoch_nb += 0.5 # Note: For some reason on_epoch_end is always called twice...
        if self.shuffle_data:
            for i in range(len(self.doc_ids)):
                if self.data_group_size == 1:
                    shuffle(self.doc_ids[i])
                else:
                    # we made sure that len(self.doc_ids) is a multiple of self.docs_per_batch in the constructor
                    x = np.reshape(self.doc_ids[i], (-1, self.docs_per_batch))
                    np.random.shuffle(x)
                    self.doc_ids[i] = x.flatten().tolist()
