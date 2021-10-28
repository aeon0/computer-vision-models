from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import urllib.parse
from typing import NamedTuple, List, NoReturn
from configparser import ConfigParser
import configparser


class MongoDBConnectionConfig(NamedTuple):
    name: str
    url: str
    port: int
    user: str = None
    pwd: str = None
    client: MongoClient = None


class MongoDBConnect:
    """
    Class to store and manage mongoDB connections
    """
    def __init__(self):
        self._connections: List[MongoDBConnectionConfig] = []

    def add_connections_from_file(self, file_name) -> NoReturn:
        """
        Add connections from config file
        :param file_name: path to config file
        """
        cp = configparser.ConfigParser()
        if len(cp.read(file_name)) == 0:
            raise ValueError("Config File could not be loaded, please check the correct path!")
        self.add_connections_from_config(cp)

    def add_connection(self, config: MongoDBConnectionConfig) -> NoReturn:
        """
        Adds a MongoDBConnectionConfig to the connection dict
        :param config: config that should be added of type MongoDBConnectionConfig
        """
        self._connections.append(config)

    def add_connections_from_config(self, config_parser: ConfigParser) -> NoReturn:
        """
        Takes a parsed .ini file as argument and adds all connections with type=MongoDB,
        Each section (= name) must have url, port and can have pwd and user
        :param config_parser: A ConfigParser of a .ini file
        """
        for key in config_parser.sections():
            if "db_type" in config_parser[key] and config_parser[key]["db_type"] == "MongoDB":
                self.add_connection(MongoDBConnectionConfig(
                    name=key,
                    url=config_parser[key]["url"],
                    port=int(config_parser[key]["port"]),
                    pwd=config_parser[key].get("pwd"),
                    user=config_parser[key].get("user")
                ))

    def get_client(self, name: str) -> MongoClient:
        """
        Get config data by name, connects the client if there is no prior MongoDB connection
        :param name: name of the connection config
        :return: MongoClient of the connection found for the name
        """
        con, i = self.get_connection_by_name(name)
        if con.client is None:
            con = self.connect_to(name)
        return con.client

    def get_db(self, name: str, db_name: str) -> Database:
        """
        Get mongoDB database by config name and database name,
        connects the client if there is no prior MongoDB connection
        :param name: name of the connection config
        :param db_name: name of the database
        :return: MongoDB database for the specified parameters
        """
        client = self.get_client(name)
        return client[db_name]

    def get_collection(self, name: str, db_name: str, collection: str) -> Collection:
        """
        Get collection, connects the client if there is no prior MongoDB connection
        :param name: name of the connection config
        :param db_name: name of the database
        :param collection: name of the collection
        :return: MongoDB collection for the specified parameters
        """
        db = self.get_db(name, db_name)
        return db[collection]

    def connect_to(self, name: str) -> MongoDBConnectionConfig:
        """
        Connect to connection which was previously added by its name
        :param name: Key of the connection config as string
        :return: The MongoDBConnectionConfig which the connection is to
        """
        con, i = self.get_connection_by_name(name)

        host = con.url + ":" + str(con.port)
        if con.user is not None and con.pwd is not None:
            user = urllib.parse.quote_plus(con.user)
            pwd = urllib.parse.quote_plus(con.pwd)
            con_string = 'mongodb://%s:%s@%s' % (user, pwd, host)
        else:
            con_string = 'mongodb://%s' % host

        db_client = MongoClient(con_string)
        new_con = con._replace(client=db_client)
        self._connections[i] = new_con
        return new_con

    def close_connection(self, name: str) -> NoReturn:
        """
        Close a single connection based on its config name
        :param name: name of the connection config
        """
        con, i = self.get_connection_by_name(name)
        if con.client is not None:
            con.client.close()
            con.client = None

    def remove_connection(self, name: str) -> NoReturn:
        """
        Remove a connection config by its name, closes the connection before
        :param name: name of the connection config
        """
        self.close_connection(name)
        con, i = self.get_connection_by_name(name)
        del self._connections[i]

    def get_connection_by_name(self, name: str) -> (MongoDBConnectionConfig, int):
        """
        Get connection config by its name, does not connect the client in the process!
        In case the connection name does not exist as ValueError is raised
        :param name: name of the connection config
        :return: NamedTuple of MongoDBConnectionConfig which includes all the configuration and MongoDB connection
        """
        for i, con in enumerate(self._connections):
            if con.name == name:
                return con, i
        raise ValueError(name + ": Connection does not exist!")

    def reset_connections(self) -> NoReturn:
        """ Close all added connections """
        for con in self._connections:
            if con.client is not None:
                con.client.close()
        self._connections = []
