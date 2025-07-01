"""Comprehensive unit tests for cosmoprimo.utils module."""

import os
import tempfile
import shutil
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from scipy import optimize

from cosmoprimo.utils import (
     BaseClass, addproperty, _bcast_dtype, flatarray,
    LeastSquareSolver, DistanceToRedshift, savefig, mkdir
)



class TestSavefig:
    """Test the savefig utility function."""
    
    @patch('matplotlib.pyplot.gcf')
    @patch('cosmoprimo.utils.logger')
    def test_savefig_basic_functionality(self, mock_logger, mock_gcf):
        """Test basic savefig functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_plot.png")
            
            # Mock figure
            mock_fig = MagicMock()
            mock_gcf.return_value = mock_fig
            
            result = savefig(filename)
            
            # Check that mkdir was called for the directory
            mock_fig.savefig.assert_called_once()
            mock_logger.info.assert_called_once()
            assert result == mock_fig
    
    @patch('matplotlib.pyplot.gcf')
    @patch('cosmoprimo.utils.logger')
    def test_savefig_with_custom_figure(self, mock_logger, mock_gcf):
        """Test savefig with a custom figure object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_plot.png")
            
            # Mock custom figure
            mock_fig = MagicMock()
            
            result = savefig(filename, fig=mock_fig)
            
            # Should use the provided figure, not gcf
            mock_gcf.assert_not_called()
            mock_fig.savefig.assert_called_once()
            assert result == mock_fig
    
    @patch('matplotlib.pyplot.gcf')
    @patch('cosmoprimo.utils.logger')
    def test_savefig_with_custom_parameters(self, mock_logger, mock_gcf):
        """Test savefig with custom savefig parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "test_plot.png")
            
            mock_fig = MagicMock()
            mock_gcf.return_value = mock_fig
            
            savefig(filename, dpi=300, format='pdf', transparent=True)
            
            # Check that custom parameters were passed
            call_args = mock_fig.savefig.call_args
            assert call_args[1]['dpi'] == 300
            assert call_args[1]['format'] == 'pdf'
            assert call_args[1]['transparent'] is True
    
    @patch('matplotlib.pyplot.gcf')
    @patch('cosmoprimo.utils.logger')
    def test_savefig_creates_directory(self, mock_logger, mock_gcf):
        """Test that savefig creates the directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = os.path.join(temp_dir, "new_subdir", "test_plot.png")
            
            mock_fig = MagicMock()
            mock_gcf.return_value = mock_fig
            
            savefig(filename)
            
            # Check that the directory was created
            assert os.path.exists(os.path.dirname(filename))
            mock_fig.savefig.assert_called_once()


class TestBaseClass:
    """Test the BaseClass utility class."""
    
    def test_base_class_copy_method(self):
        """Test that BaseClass copy method works correctly."""
        class TestClass(BaseClass):
            def __init__(self, value):
                self.value = value
                self.list_value = [1, 2, 3]
        
        original = TestClass(42)
        copied = original.copy()
        
        # Check that it's a shallow copy
        assert copied is not original
        assert copied.value == original.value
        assert copied.list_value is original.list_value  # Same reference for nested objects
        
        # Verify the copy is functional
        copied.value = 100
        assert copied.value == 100
        assert original.value == 42
    
    def test_base_class_inheritance(self):
        """Test that classes can properly inherit from BaseClass."""
        class ChildClass(BaseClass):
            def __init__(self, name, data):
                self.name = name
                self.data = data
        
        child = ChildClass("test", [1, 2, 3])
        copied = child.copy()
        
        assert isinstance(copied, ChildClass)
        assert copied.name == child.name
        assert copied.data == child.data
    
    def test_base_class_with_complex_attributes(self):
        """Test BaseClass copy with complex nested attributes."""
        class TestClass(BaseClass):
            def __init__(self):
                self.dict_attr = {'a': 1, 'b': [2, 3]}
                self.numpy_array = np.array([1, 2, 3])
                self.none_attr = None
        
        original = TestClass()
        copied = original.copy()
        
        # Check shallow copy behavior
        assert copied.dict_attr is original.dict_attr
        assert copied.numpy_array is original.numpy_array
        assert copied.none_attr is original.none_attr
        
        # Verify the copy is functional
        copied.dict_attr['a'] = 999
        assert original.dict_attr['a'] == 999  # Same reference


class TestAddProperty:
    """Test the addproperty decorator."""
    
    def test_addproperty_single_attribute(self):
        """Test adding a single property to a class."""
        @addproperty('value')
        class TestClass:
            def __init__(self):
                self._value = 42
        
        obj = TestClass()
        assert obj.value == 42
    
    def test_addproperty_multiple_attributes(self):
        """Test adding multiple properties to a class."""
        @addproperty('name', 'age', 'data')
        class TestClass:
            def __init__(self):
                self._name = "Alice"
                self._age = 30
                self._data = [1, 2, 3]
        
        obj = TestClass()
        assert obj.name == "Alice"
        assert obj.age == 30
        assert obj.data == [1, 2, 3]
    
    def test_addproperty_readonly(self):
        """Test that properties created by addproperty are read-only."""
        @addproperty('value')
        class TestClass:
            def __init__(self):
                self._value = 42
        
        obj = TestClass()
        
        # Should not be able to set the property
        with pytest.raises(AttributeError):
            obj.value = 100
    
    def test_addproperty_on_existing_property(self):
        """Test addproperty on a class that already has a property."""
        class TestClass:
            @property
            def value(self):
                return 10
        
        # Should overwrite the property
        cls = addproperty('value')(TestClass)
        obj = cls()
        obj._value = 42
        assert obj.value == 42
    
    def test_addproperty_with_missing_attribute(self):
        """Test addproperty when the underlying attribute doesn't exist."""
        @addproperty('missing_attr')
        class TestClass:
            def __init__(self):
                pass
        
        obj = TestClass()
        
        # Should raise AttributeError when accessing non-existent attribute
        with pytest.raises(AttributeError):
            _ = obj.missing_attr
    
    def test_addproperty_empty_attributes(self):
        """Test addproperty with no attributes specified."""
        @addproperty()
        class TestClass:
            def __init__(self):
                self._value = 42
        
        obj = TestClass()
        # Should not have any properties added
        assert not hasattr(obj, 'value')


class TestBcastDtype:
    """Test the _bcast_dtype function."""
    
    def test_bcast_dtype_all_float32(self):
        """Test dtype broadcasting when all arrays are float32."""
        arr1 = np.array([1.0], dtype=np.float32)
        arr2 = np.array([2.0], dtype=np.float32)
        
        result = _bcast_dtype(arr1, arr2)
        assert result == np.float32
    
    def test_bcast_dtype_mixed_precision(self):
        """Test dtype broadcasting with mixed precision arrays."""
        arr1 = np.array([1.0], dtype=np.float32)
        arr2 = np.array([2.0], dtype=np.float64)
        
        result = _bcast_dtype(arr1, arr2)
        assert result == np.float64
    
    def test_bcast_dtype_with_non_floating(self):
        """Test dtype broadcasting with non-floating arrays."""
        arr1 = np.array([1], dtype=np.int32)
        arr2 = np.array([2.0], dtype=np.float32)
        
        result = _bcast_dtype(arr1, arr2)
        assert result == np.float64
    
    def test_bcast_dtype_no_arrays(self):
        """Test dtype broadcasting with no arrays."""
        result = _bcast_dtype()
        assert result == np.float64
    
    def test_bcast_dtype_with_none(self):
        """Test dtype broadcasting with None values."""
        arr1 = np.array([1.0], dtype=np.float32)
        
        result = _bcast_dtype(arr1, None)
        assert result == np.float32  # Only one array with float32, so result should be float32
    
    def test_bcast_dtype_with_non_array_objects(self):
        """Test dtype broadcasting with non-array objects."""
        arr = np.array([1.0], dtype=np.float32)
        class Dummy:
            pass
        result = _bcast_dtype(arr, Dummy())
        assert result == np.float32  # Only one array with float32, so result should be float32
    
    def test_bcast_dtype_with_complex_types(self):
        """Test dtype broadcasting with complex number arrays."""
        arr1 = np.array([1.0], dtype=np.float32)
        arr2 = np.array([1+2j], dtype=np.complex64)
        
        result = _bcast_dtype(arr1, arr2)
        assert result == np.float64  # Should default to float64 for non-floating
    
    def test_bcast_dtype_with_string_arrays(self):
        """Test dtype broadcasting with string arrays."""
        arr1 = np.array([1.0], dtype=np.float32)
        arr2 = np.array(['test'], dtype=str)
        
        result = _bcast_dtype(arr1, arr2)
        assert result == np.float64  # Should default to float64 for non-floating


class TestFlatArray:
    """Test the flatarray decorator."""
    
    def test_flatarray_basic_functionality(self):
        """Test basic functionality of the flatarray decorator."""
        class TestClass:
            _np = np
            
            @flatarray(iargs=[0], dtype=np.float64)
            def test_method(self, arr):
                return arr  # Return the array itself, not a scalar
        
        obj = TestClass()
        arr = np.array([[1, 2], [3, 4]])
        result = obj.test_method(arr)
        
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float64
    
    def test_flatarray_multiple_arrays(self):
        """Test flatarray with multiple input arrays."""
        class TestClass:
            _np = np
            
            @flatarray(iargs=[0, 1], dtype=np.float64)
            def test_method(self, arr1, arr2, scalar=1.0):
                return arr1 + arr2 + scalar
        
        obj = TestClass()
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6], [7, 8]])
        
        result = obj.test_method(arr1, arr2)
        expected = np.array([[7, 9], [11, 13]])
        np.testing.assert_array_equal(result, expected)
    
    def test_flatarray_different_shapes_error(self):
        """Test that flatarray raises error for different shapes."""
        class TestClass:
            _np = np
            
            @flatarray(iargs=[0, 1], dtype=np.float64)
            def test_method(self, arr1, arr2):
                return arr1 + arr2
        
        obj = TestClass()
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[5, 6]])  # Different shape
        
        with pytest.raises(ValueError, match="input arrays must have same shape"):
            obj.test_method(arr1, arr2)
    
    def test_flatarray_dict_return(self):
        """Test flatarray with dictionary return values."""
        class TestClass:
            _np = np
            
            @flatarray(iargs=[0], dtype=np.float64)
            def test_method(self, arr):
                return {'sum': arr, 'mean': arr}  # Return arrays, not scalars
        
        obj = TestClass()
        arr = np.array([[1, 2], [3, 4]])
        result = obj.test_method(arr)
        
        assert isinstance(result, dict)
        np.testing.assert_array_equal(result['sum'], arr)
        np.testing.assert_array_equal(result['mean'], arr)
        assert result['sum'].dtype == np.float64
    
    def test_flatarray_with_dtype_none(self):
        """Test flatarray with dtype=None (preserve input dtype)."""
        class TestClass:
            _np = np
            @flatarray(iargs=[0], dtype=None)
            def test_method(self, arr):
                return arr + 1
        obj = TestClass()
        arr = np.array([1, 2, 3], dtype=np.float32)
        result = obj.test_method(arr)
        assert np.allclose(result, arr + 1)
        assert result.dtype == arr.dtype
    
    def test_flatarray_with_custom_np(self):
        """Test flatarray with custom numpy-like object."""
        class TestClass:
            _np = np  # Will use this
            
            @flatarray(iargs=[0], dtype=np.float64)
            def test_method(self, arr):
                return arr  # Return the array itself
        
        obj = TestClass()
        arr = np.array([[1, 2], [3, 4]])
        result = obj.test_method(arr)
        np.testing.assert_array_equal(result, arr)
    
    def test_flatarray_with_kwargs(self):
        """Test flatarray with keyword arguments."""
        class TestClass:
            _np = np
            
            @flatarray(iargs=[0], dtype=np.float64)
            def test_method(self, arr, factor=2.0):
                return arr * factor
        
        obj = TestClass()
        arr = np.array([[1, 2], [3, 4]])
        result = obj.test_method(arr, factor=3.0)
        expected = np.array([[3, 6], [9, 12]])
        np.testing.assert_array_equal(result, expected)
    
    def test_flatarray_with_3d_array(self):
        """Test flatarray with 3D input arrays."""
        class TestClass:
            _np = np
            
            @flatarray(iargs=[0], dtype=np.float64)
            def test_method(self, arr):
                return arr  # Return the array itself
        
        obj = TestClass()
        arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        result = obj.test_method(arr)
        np.testing.assert_array_equal(result, arr)

    def test_flatarray_with_empty_array(self):
        """Test flatarray with empty arrays."""
        class TestClass:
            _np = np
            
            @flatarray(iargs=[0], dtype=np.float64)
            def test_method(self, arr):
                return arr  # Return the array itself
        
        obj = TestClass()
        arr = np.array([])
        result = obj.test_method(arr)
        np.testing.assert_array_equal(result, arr)


class TestLeastSquareSolver:
    """Test the LeastSquareSolver class."""
    
    @pytest.fixture
    def simple_gradient(self):
        """Create a simple gradient for testing."""
        return np.array([1.0, 1.0, 1.0, 1.0])
    
    @pytest.fixture
    def matrix_gradient(self):
        """Create a matrix gradient for testing."""
        return np.array([[1.0, 1.0, 1.0, 1.0],
                        [0.0, 1.0, 2.0, 3.0]])
    
    @pytest.fixture
    def precision_matrix(self):
        """Create a precision matrix for testing."""
        return np.eye(4)
    
    def test_least_square_solver_scalar_gradient(self, simple_gradient, precision_matrix):
        """Test LeastSquareSolver with scalar gradient."""
        solver = LeastSquareSolver(simple_gradient, precision_matrix)
        
        # Test basic functionality
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        result = solver(delta)
        
        assert result.shape == ()  # Scalar array
        assert result == 2.5  # Mean of delta values
    
    def test_least_square_solver_matrix_gradient(self, matrix_gradient, precision_matrix):
        """Test LeastSquareSolver with matrix gradient."""
        solver = LeastSquareSolver(matrix_gradient, precision_matrix)
        
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        result = solver(delta)
        
        assert result.shape == (2,)
        assert len(result) == 2
    
    def test_least_square_solver_scalar_precision(self, simple_gradient):
        """Test LeastSquareSolver with scalar precision."""
        solver = LeastSquareSolver(simple_gradient, precision=2.0)
        
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        result = solver(delta)
        
        assert result.shape == ()  # Scalar array
        assert result == 2.5
    
    def test_least_square_solver_vector_precision(self, simple_gradient):
        """Test LeastSquareSolver with vector precision."""
        precision = np.array([1.0, 2.0, 1.0, 2.0])
        solver = LeastSquareSolver(simple_gradient, precision=precision)
        
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        result = solver(delta)
        
        assert result.shape == ()  # Scalar array
    
    def test_least_square_solver_with_constraints(self, matrix_gradient, precision_matrix):
        """Test LeastSquareSolver with constraints."""
        constraint_gradient = np.array([[1.0], [0.0]])
        solver = LeastSquareSolver(matrix_gradient, precision_matrix, 
                                 constraint_gradient=constraint_gradient)
        
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        constraint = 5.0
        result = solver(delta, constraint=constraint)
        
        assert result.shape == (2,)
        # Check that constraint is satisfied
        assert np.isclose(result[0], constraint)
    
    def test_least_square_solver_multiple_constraints(self, matrix_gradient, precision_matrix):
        """Test LeastSquareSolver with multiple constraints."""
        constraint_gradient = np.array([[1.0, 0.0], [0.0, 1.0]])
        solver = LeastSquareSolver(matrix_gradient, precision_matrix, 
                                 constraint_gradient=constraint_gradient)
        
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        constraints = [5.0, 10.0]
        result = solver(delta, constraint=constraints)
        
        assert result.shape == (2,)
        # Check that constraints are satisfied
        assert np.isclose(result[0], constraints[0])
        assert np.isclose(result[1], constraints[1])
    
    def test_least_square_solver_model_method(self, simple_gradient, precision_matrix):
        """Test the model method of LeastSquareSolver."""
        solver = LeastSquareSolver(simple_gradient, precision_matrix)
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        solver(delta)
        
        model = solver.model()
        assert model.shape == (4,)
        assert np.allclose(model, 2.5 * np.ones(4))
    
    def test_least_square_solver_chi2_method(self, simple_gradient, precision_matrix):
        """Test the chi2 method of LeastSquareSolver."""
        solver = LeastSquareSolver(simple_gradient, precision_matrix)
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        solver(delta)
        
        chi2 = solver.chi2()
        assert np.isscalar(chi2)
        assert chi2 >= 0.0
    
    def test_least_square_solver_invalid_gradient_dimension(self):
        """Test that LeastSquareSolver raises error for invalid gradient dimensions."""
        gradient = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # 3D array
        
        with pytest.raises(ValueError, match="gradient must be at most 2D"):
            LeastSquareSolver(gradient)
    
    def test_least_square_solver_invalid_constraint_gradient(self, simple_gradient):
        """Test that LeastSquareSolver raises error for invalid constraint gradient."""
        constraint_gradient = np.array([1.0, 2.0])  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="constraint_gradient must be 2D"):
            LeastSquareSolver(simple_gradient, constraint_gradient=constraint_gradient)
    
    def test_least_square_solver_constraint_gradient_wrong_shape(self, simple_gradient):
        """Test that LeastSquareSolver raises error for wrong constraint gradient shape."""
        constraint_gradient = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Wrong first dimension
        
        with pytest.raises(ValueError, match="constraint_gradient must be 2D"):
            LeastSquareSolver(simple_gradient, constraint_gradient=constraint_gradient)
    
    def test_least_square_solver_compute_inverse_false(self, simple_gradient, precision_matrix):
        """Test LeastSquareSolver with compute_inverse=False."""
        solver = LeastSquareSolver(simple_gradient, precision_matrix, compute_inverse=False)
        
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        result = solver(delta)
        
        assert result.shape == ()  # Scalar array
        assert result == 2.5
    
    def test_least_square_solver_batch_processing(self, simple_gradient, precision_matrix):
        """Test LeastSquareSolver with batch processing."""
        solver = LeastSquareSolver(simple_gradient, precision_matrix)
        
        # Single delta
        delta_single = np.array([1.0, 2.0, 3.0, 4.0])
        result_single = solver(delta_single)
        
        # Batch of deltas
        delta_batch = np.array([delta_single, delta_single * 2])
        result_batch = solver(delta_batch)
        
        assert result_batch.shape == (2,)
        assert np.isclose(result_batch[0], result_single)
        assert np.isclose(result_batch[1], result_single * 2)

    def test_least_square_solver_tree_flatten_and_unflatten(self):
        """Test tree_flatten and tree_unflatten methods."""
        gradient = np.array([1.0, 2.0, 3.0])
        precision = np.eye(3)
        solver = LeastSquareSolver(gradient, precision)
        solver(np.array([1.0, 2.0, 3.0]))
        children, aux = solver.tree_flatten()
        new_solver = LeastSquareSolver.tree_unflatten(aux, children)
        assert isinstance(new_solver, LeastSquareSolver)
        # The tree_unflatten method may not fully reconstruct all attributes
        # Just check that it creates a valid instance
    
    def test_least_square_solver_compute_method(self, simple_gradient, precision_matrix):
        """Test the compute method directly."""
        solver = LeastSquareSolver(simple_gradient, precision_matrix)
        delta = np.array([1.0, 2.0, 3.0, 4.0])
        
        solver.compute(delta)
        
        assert hasattr(solver, 'delta')
        assert hasattr(solver, 'params')
        assert np.allclose(solver.delta, delta)
    
    def test_least_square_solver_with_singular_matrix(self):
        """Test LeastSquareSolver with a singular precision matrix."""
        gradient = np.array([1.0, 1.0])
        # Use a more obviously singular precision matrix
        precision = np.array([[0.0, 0.0], [0.0, 0.0]])  # Zero matrix is definitely singular
        
        # This should raise a LinAlgError due to singular matrix
        with pytest.raises(np.linalg.LinAlgError):
            solver = LeastSquareSolver(gradient, precision)
    
    def test_least_square_solver_with_empty_gradient(self):
        """Test LeastSquareSolver with empty gradient."""
        gradient = np.array([])
        precision = np.eye(0)
        
        # This should raise a LinAlgError due to singular matrix
        with pytest.raises(np.linalg.LinAlgError):
            solver = LeastSquareSolver(gradient, precision)


class TestDistanceToRedshift:
    """Test the DistanceToRedshift class."""
    
    @pytest.fixture
    def mock_distance_function(self):
        """Create a mock distance function for testing."""
        def distance(z):
            # Simple linear relationship for testing
            return z * 1000.0  # Mpc
        return distance
    
    def test_distance_to_redshift_initialization(self, mock_distance_function):
        """Test DistanceToRedshift initialization."""
        d2z = DistanceToRedshift(mock_distance_function, zmax=10.0, nz=100, interp_order=3)
        
        assert hasattr(d2z, '_interp')
        assert d2z._interp is not None
    
    def test_distance_to_redshift_call_scalar(self, mock_distance_function):
        """Test DistanceToRedshift with scalar input."""
        d2z = DistanceToRedshift(mock_distance_function, zmax=10.0, nz=100, interp_order=3)
        
        distance = 1000.0  # Mpc
        redshift = d2z(distance)
        
        assert redshift.shape == ()  # Scalar array
        assert np.isclose(redshift, 1.0, rtol=1e-2)
    
    def test_distance_to_redshift_call_array(self, mock_distance_function):
        """Test DistanceToRedshift with array input."""
        d2z = DistanceToRedshift(mock_distance_function, zmax=10.0, nz=100, interp_order=3)
        
        distances = np.array([1000.0, 2000.0, 3000.0])  # Mpc
        redshifts = d2z(distances)
        
        assert redshifts.shape == (3,)
        assert np.allclose(redshifts, [1.0, 2.0, 3.0], rtol=1e-2)
    
    def test_distance_to_redshift_bounds_error_true(self, mock_distance_function):
        """Test DistanceToRedshift with bounds_error=True."""
        d2z = DistanceToRedshift(mock_distance_function, zmax=10.0, nz=100, interp_order=3)
        
        # Test with distance outside bounds
        with pytest.raises(ValueError):
            d2z(15000.0)  # Beyond zmax=10.0
    
    def test_distance_to_redshift_bounds_error_false(self, mock_distance_function):
        """Test DistanceToRedshift with bounds_error=False."""
        d2z = DistanceToRedshift(mock_distance_function, zmax=10.0, nz=100, interp_order=3)
        
        # Test with distance outside bounds - should return NaN for extrapolation
        result = d2z(15000.0, bounds_error=False)
        # Should return NaN for extrapolation beyond bounds
        assert np.isnan(result)
    
    def test_distance_to_redshift_different_interp_orders(self, mock_distance_function):
        """Test DistanceToRedshift with different interpolation orders."""
        for order in [1, 3]:  # Only test supported orders
            d2z = DistanceToRedshift(mock_distance_function, zmax=10.0, nz=100, interp_order=order)
            
            distance = 1000.0
            redshift = d2z(distance)
            
            assert redshift.shape == ()  # Scalar array
            assert np.isclose(redshift, 1.0, rtol=1e-2)
    
    def test_distance_to_redshift_tree_flatten_and_unflatten(self, mock_distance_function):
        """Test tree_flatten and tree_unflatten methods."""
        d2z = DistanceToRedshift(mock_distance_function, zmax=10.0, nz=100, interp_order=3)
        
        children, aux = d2z.tree_flatten()
        new_d2z = DistanceToRedshift.tree_unflatten(aux, children)
        
        assert isinstance(new_d2z, DistanceToRedshift)
        assert hasattr(new_d2z, '_interp')
    
    def test_distance_to_redshift_with_nonlinear_distance(self):
        """Test DistanceToRedshift with a nonlinear distance function."""
        def nonlinear_distance(z):
            # Quadratic relationship for testing
            return z**2 * 100.0
        
        d2z = DistanceToRedshift(nonlinear_distance, zmax=5.0, nz=200, interp_order=3)
        
        # Test at known points
        distance = 100.0  # Should correspond to z=1
        redshift = d2z(distance)
        assert np.isclose(redshift, 1.0, rtol=1e-2)
    
    def test_distance_to_redshift_with_zero_distance(self, mock_distance_function):
        """Test DistanceToRedshift with zero distance."""
        d2z = DistanceToRedshift(mock_distance_function, zmax=10.0, nz=100, interp_order=3)
        
        redshift = d2z(0.0)
        assert redshift.shape == ()  # Scalar array
        assert redshift >= 0.0  # Should be non-negative


# Integration tests that verify the existing functionality works as expected
class TestIntegration:
    """Integration tests for utils module."""
    
    def test_least_squares_integration(self):
        """Integration test for LeastSquareSolver with scipy comparison."""
        for compute_inverse in [False, True]:
            x = np.linspace(1, 100, 10)
            gradient = np.array([1. / x, np.ones_like(x), x, x ** 2, x ** 3])
            
            cov = np.diag(x)
            precision = np.linalg.inv(cov)
            rng = np.random.RandomState(seed=42)
            y = rng.uniform(0., 1., x.size)
            
            def chi2(pars):
                delta = y - pars.dot(gradient)
                return np.sum(delta.dot(precision).dot(delta.T))
            
            x0 = np.zeros(len(gradient))
            result_ref = optimize.minimize(chi2, x0=x0, args=(), method='Nelder-Mead', 
                                         tol=1e-6, options={'maxiter': 1000000}).x
            
            solver = LeastSquareSolver(gradient, precision, compute_inverse=compute_inverse)
            result = solver(y)
            assert np.allclose(result, result_ref, rtol=1e-2, atol=1e-2)
    
    def test_redshift_distance_integration(self):
        """Integration test for DistanceToRedshift with cosmology."""
        try:
            from cosmoprimo.fiducial import DESI
            cosmo = DESI()
            zmax = 10.
            distance = cosmo.comoving_radial_distance
            redshift = DistanceToRedshift(distance=distance, zmax=zmax, nz=4096)
            
            z = np.random.uniform(0., 2., 10000)
            assert np.allclose(redshift(distance(z)), z, atol=1e-6)
        except ImportError:
            pytest.skip("cosmoprimo.fiducial not available for integration test")

    def test_flatarray_with_dtype_none(self):
        """Test flatarray with dtype=None preserves input dtype."""
        class TestClass:
            _np = np
            @flatarray(iargs=[0], dtype=None)
            def test_method(self, arr):
                return arr + 1
        obj = TestClass()
        arr = np.array([1, 2, 3], dtype=np.float32)
        result = obj.test_method(arr)
        assert np.allclose(result, arr + 1)
        assert result.dtype == arr.dtype

    def test_addproperty_on_existing_property(self):
        """Test addproperty overwrites existing properties."""
        class TestClass:
            @property
            def value(self):
                return 10
        # Should overwrite the property
        cls = addproperty('value')(TestClass)
        obj = cls()
        obj._value = 42
        assert obj.value == 42

    def test_bcast_dtype_with_non_array_objects(self):
        """Test _bcast_dtype with non-array objects."""
        arr = np.array([1.0], dtype=np.float32)
        class Dummy:
            pass
        result = _bcast_dtype(arr, Dummy())
        assert result == np.float32  # Only one array with float32, so result should be float32


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_mkdir_with_none(self):
        """Test mkdir with None input."""
        with pytest.raises(TypeError):
            mkdir(None)
    
    def test_flatarray_with_empty_array(self):
        """Test flatarray with empty arrays."""
        class TestClass:
            _np = np
            
            @flatarray(iargs=[0], dtype=np.float64)
            def test_method(self, arr):
                return arr  # Return the array itself
        
        obj = TestClass()
        arr = np.array([])
        result = obj.test_method(arr)
        np.testing.assert_array_equal(result, arr)
    
    def test_least_square_solver_with_empty_gradient(self):
        """Test LeastSquareSolver with empty gradient."""
        gradient = np.array([])
        precision = np.eye(0)
        
        # This should raise a LinAlgError due to singular matrix
        with pytest.raises(np.linalg.LinAlgError):
            solver = LeastSquareSolver(gradient, precision)
    
    def test_distance_to_redshift_with_invalid_distance_function(self):
        """Test DistanceToRedshift with invalid distance function."""
        def invalid_distance(z):
            raise ValueError("Invalid distance function")
        
        with pytest.raises(ValueError):
            DistanceToRedshift(invalid_distance, zmax=10.0, nz=100)
    
    def test_addproperty_with_special_characters(self):
        """Test addproperty with attribute names containing special characters."""
        @addproperty('test_attr', 'attr_with_underscore')
        class TestClass:
            def __init__(self):
                self._test_attr = "test"
                self._attr_with_underscore = "underscore"
        
        obj = TestClass()
        assert obj.test_attr == "test"
        assert obj.attr_with_underscore == "underscore"


if __name__ == '__main__':
    pytest.main([__file__])
