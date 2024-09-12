## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = cosine_similarity(vector1, vector2)
    
    dot_prod = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    expected_result = dot_prod / (norm_vector1 * norm_vector2)
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    dataset = np.array([[1, 2], [3, 4], [5, 6]])
    query_point = np.array([2, 3])
    
    result = nearest_neighbor(dataset, query_point)
    
    distances = np.linalg.norm(dataset - query_point, axis=1)
    expected_index = np.argmin(distances)
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
