# code for estimator.py
import os
import sys
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import referal_pred_Exception
from src.logger import logging

class TargetValueMapping:
   def __init__(self):
       self.no:int = 0
       self.yes:int = 1
   def _asdict(self):
       return self.__dict__
   def reverse_mapping(self):
       mapping_response = self._asdict()
       return dict(zip(mapping_response.values(),mapping_response.keys()))