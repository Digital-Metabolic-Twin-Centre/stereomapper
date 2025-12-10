import os
import sys
import warnings
import rdkit
from rdkit import RDLogger
from contextlib import contextmanager

def setup_clean_logging():
    """Configure clean logging for pipeline execution"""
    
    # Suppress RDKit warnings
    RDLogger.DisableLog('rdApp.*')
    
    # Suppress Python warnings related to chemistry libraries
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Suppress specific InChI warnings
    warnings.filterwarnings('ignore', message='.*InChI.*')

@contextmanager
def quiet_operation():
    """Context manager for completely silent operations"""
    with open('/dev/null', 'w') as devnull:
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        try:
            sys.stderr = devnull
            if os.getenv('QUIET_MODE', 'false').lower() == 'true':
                sys.stdout = devnull
            yield
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout