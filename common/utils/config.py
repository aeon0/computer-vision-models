import configparser


class Config:
    """
    Static class to manage config data from a config.ini file
    """
    _config_parser: configparser.ConfigParser

    @staticmethod
    def add_config(config_path):
        """
        Add a config.ini file and save its contents to the Config instance
        :param config_path: path to the config.ini file
        """
        cp = configparser.ConfigParser()
        cp.read(config_path)
        Config._config_parser = cp

    @staticmethod
    def get_config_parser() -> configparser.ConfigParser:
        """
        :return: return config parser of config.ini file
        """
        return Config._config_parser
