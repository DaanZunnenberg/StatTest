import importlib
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_function(module) -> ...: print(f"Initializing module: {module.__name__}")

def cleanup_function(module) -> ...: print(f"Cleaning up module: {module.__name__}")

class ImportContextManager:
    def __init__(self, module_name, init_func=None, cleanup_func=None):
        """
        Initialize the context manager.

        :param module_name: Name of the module to import.
        :param init_func: Optional function to call after importing the module.
        :param cleanup_func: Optional function to call before cleaning up the module.
        """
        self.module_name = module_name
        self.module = None
        self.init_func = init_func
        self.cleanup_func = cleanup_func

    def __enter__(self):
        try:
            logger.info(f"Importing module {self.module_name}")
            self.module = importlib.import_module(self.module_name)
            if self.init_func:
                logger.info(f"Running initialization function for module {self.module_name}")
                self.init_func(self.module)
            return self.module
        except ImportError as e:
            logger.error(f"Failed to import module {self.module_name}: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_func:
            try:
                logger.info(f"Running cleanup function for module {self.module_name}")
                self.cleanup_func(self.module)
            except Exception as e:
                logger.error(f"Error during cleanup of module {self.module_name}: {e}")
                raise
        if self.module_name in sys.modules:
            logger.info(f"Removing module {self.module_name} from sys.modules")
            del sys.modules[self.module_name]
        if exc_type:
            logger.error(f"Exception occurred: {exc_type}, {exc_val}")
