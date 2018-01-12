"""bundles all the pytest calls for the code"""
import pytest
if __name__ == '__main__':
    pytest.cmdline.main(['algebra_tests.py'])
    pytest.cmdline.main(['fisher_tests.py'])
    pytest.cmdline.main(['power_comparison_tests.py'])
    pytest.cmdline.main(['power_derivative_tests.py'])
    pytest.cmdline.main(['power_response_tests.py'])
    pytest.cmdline.main(['spherical_harmonic_tests.py'])
    pytest.cmdline.main(['sph_tests.py'])
    pytest.cmdline.main(['w_matcher_tests.py'])
    pytest.cmdline.main(['polygon_geo_tests.py'])
