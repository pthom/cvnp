import pytest
import numpy as np
import math
import random
import pytest

import sys
sys.path.append(".")
sys.path.append("..")


from cvnp import short_lived_matx, short_lived_mat


def are_float_close(x: float, y: float):
    return math.fabs(x - y) < 1E-5


def test_short_lived_matx():
    """
    We are calling the function ShortLivedMatx():

        // Returns a short lived matrix: sharing memory for this matrix makes *no sense at all*,
        // since its pointer lives on the stack and is deleted as soon as we exit the function!
        cv::Matx33d ShortLivedMatx()
        {
            auto mat = cv::Matx33d::eye();
            return mat;
        }
    """
    m = short_lived_matx()
    assert are_float_close(m[0, 0], 1.)


def test_short_lived_mat():
    """
    We are calling the function ShortLivedMat():

        // Returns a short lived Mat: sharing memory for this matrix makes *no sense at all*,
        // since its pointer lives on the stack and is deleted as soon as we exit the function!
        cv::Mat ShortLivedMat()
        {
            auto mat = cv::Mat(cv::Size(300, 200), CV_8UC4);
            mat = cv::Scalar(12, 34, 56, 78);
            return mat;
        }
    """
    m = short_lived_mat()
    assert m.shape == (200, 300, 4)
    assert (m[0, 0] == (12, 34, 56, 78)).all()
