"""bundles all the pytest calls for the code"""
#other tests: cosmolike_comparision_test, projected_tests,cosmolike_comparison_test_ssc
from __future__ import absolute_import,print_function,division
from builtins import range
import pytest
if __name__=='__main__':
    pytest.cmdline.main(['algebra_tests.py'])
    pytest.cmdline.main(['fisher_tests.py'])
    pytest.cmdline.main(['extrap_util_tests.py'])
    pytest.cmdline.main(['power_comparison_tests.py'])
    pytest.cmdline.main(['power_derivative_tests.py'])
    pytest.cmdline.main(['power_response_tests.py'])
    pytest.cmdline.main(['hmf_tests.py'])
    pytest.cmdline.main(['projected_tests.py'])
    pytest.cmdline.main(['cosmolike_comparison_test.py'])
    pytest.cmdline.main(['spherical_harmonic_tests.py'])
    pytest.cmdline.main(['sph_geo_version_test.py'])
    pytest.cmdline.main(['sph_tests.py'])
    pytest.cmdline.main(['w_matcher_tests.py'])
    pytest.cmdline.main(['polygon_geo_tests.py'])
    pytest.cmdline.main(['prior_fisher_tests.py'])
    pytest.cmdline.main(['sph_basis_tests.py'])
    pytest.cmdline.main(['super_survey_test.py'])
    pytest.cmdline.main(['cosmo_response_tests.py'])
