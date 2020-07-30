"""
DBSCAN: Density-Based Spatial Clustering of Applications with Noise
"""

import numpy as np
from scipy import sparse
from _dbscan_inner import dbscan_inner
import NearestNeighbors

class BaseEstimator:
    """Base class for all estimators in scikit-learn
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """


    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self, N_CHAR_MAX=700):
        # N_CHAR_MAX is the (approximate) maximum number of non-blank
        # characters to render. We pass it as an optional parameter to ease
        # the tests.

        from .utils._pprint import _EstimatorPrettyPrinter

        N_MAX_ELEMENTS_TO_SHOW = 30  # number of elements to show in sequences

        # use ellipsis for sequences with a lot of elements
        pp = _EstimatorPrettyPrinter(
            compact=True, indent=1, indent_at_name=True,
            n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW)

        repr_ = pp.pformat(self)

        # Use bruteforce ellipsis when there are a lot of non-blank characters
        n_nonblank = len(''.join(repr_.split()))
        if n_nonblank > N_CHAR_MAX:
            lim = N_CHAR_MAX // 2  # apprx number of chars to keep on both ends
            regex = r'^(\s*\S){%d}' % lim
            # The regex '^(\s*\S){%d}' % n
            # matches from the start of the string until the nth non-blank
            # character:
            # - ^ matches the start of string
            # - (pattern){n} matches n repetitions of pattern
            # - \s*\S matches a non-blank char following zero or more blanks
            left_lim = re.match(regex, repr_).end()
            right_lim = re.match(regex, repr_[::-1]).end()

            if '\n' in repr_[left_lim:-right_lim]:
                # The left side and right side aren't on the same line.
                # To avoid weird cuts, e.g.:
                # categoric...ore',
                # we need to start the right side with an appropriate newline
                # character so that it renders properly as:
                # categoric...
                # handle_unknown='ignore',
                # so we add [^\n]*\n which matches until the next \n
                regex += r'[^\n]*\n'
                right_lim = re.match(regex, repr_[::-1]).end()

            ellipsis = '...'
            if left_lim + len(ellipsis) < len(repr_) - right_lim:
                # Only add ellipsis if it results in a shorter repr
                repr_ = repr_[:left_lim] + '...' + repr_[-right_lim:]

        return repr_

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        if type(self).__module__.startswith('sklearn.'):
            return dict(state.items(), _sklearn_version=__version__)
        else:
            return state

    def _more_tags(self):
        return _DEFAULT_TAGS

    def _get_tags(self):
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, '_more_tags'):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags

    def _check_n_features(self, X, reset):

        n_features = X.shape[1]

        if reset:
            self.n_features_in_ = n_features
        else:
            if not hasattr(self, 'n_features_in_'):
                raise RuntimeError(
                    "The reset parameter is False but there is no "
                    "n_features_in_ attribute. Is this estimator fitted?"
                )
            if n_features != self.n_features_in_:
                raise ValueError(
                    'X has {} features, but this {} is expecting {} features '
                    'as input.'.format(n_features, self.__class__.__name__,
                                       self.n_features_in_)
                )

    def _validate_data(self, X, y=None, reset=True,
                       validate_separately=False, **check_params):


        if y is None:
            if self._get_tags()['requires_y']:
                raise ValueError(
                    f"This {self.__class__.__name__} estimator "
                    f"requires y to be passed, but the target y is None."
                )
            X = check_array(X, **check_params)
            out = X
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                X = check_array(X, **check_X_params)
                y = check_array(y, **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        if check_params.get('ensure_2d', True):
            self._check_n_features(X, reset=reset)

        return out

    def _repr_html_(self):
        """HTML representation of estimator.
        This is redundant with the logic of `_repr_mimebundle_`. The latter
        should be favorted in the long term, `_repr_html_` is only
        implemented for consumers who do not interpret `_repr_mimbundle_`.
        """
        if get_config()["display"] != 'diagram':
            raise AttributeError("_repr_html_ is only defined when the "
                                 "'display' configuration option is set to "
                                 "'diagram'")
        return self._repr_html_inner

    def _repr_html_inner(self):
        """This function is returned by the @property `_repr_html_` to make
        `hasattr(estimator, "_repr_html_") return `True` or `False` depending
        on `get_config()["display"]`.
        """
        return estimator_html_repr(self)

    def _repr_mimebundle_(self, **kwargs):
        """Mime bundle used by jupyter kernels to display estimator"""
        output = {"text/plain": repr(self)}
        if get_config()["display"] == 'diagram':
            output["text/html"] = estimator_html_repr(self)
        return output

def dbscan(X, eps=0.5, *, min_samples=5, metric='minkowski',
           metric_params=None, algorithm='auto', leaf_size=30, p=2,
           sample_weight=None, n_jobs=None):

    est = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                 metric_params=metric_params, algorithm=algorithm,
                 leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    est.fit(X, sample_weight=sample_weight)
    return est.core_sample_indices_, est.labels_


class DBSCAN(ClusterMixin, BaseEstimator):

    def __init__(self, eps=0.5, *, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):

        X = self._validate_data(X, accept_sparse='csr')

        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        neighbors_model = NearestNeighbors(
            radius=self.eps, algorithm=self.algorithm,
            leaf_size=self.leaf_size, metric=self.metric,
            metric_params=self.metric_params, p=self.p, n_jobs=self.n_jobs)
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X, return_distance=False)

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors)
                                    for neighbors in neighborhoods])
        else:
            n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                    for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples,
                                  dtype=np.uint8)
        
        # DBSCAN INNER:
        dbscan_inner(core_samples, neighborhoods, labels)

        # continue fit()

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def predict(self, X, y=None, sample_weight=None):

        #self.fit(X, sample_weight=sample_weight)
        return self.labels_


    def _check_sample_weight(self, sample_weight, X, dtype=None):

        #n_samples = _num_samples(X)

        message = 'Expected sequence or array-like, got %s' % type(X)
        if hasattr(X, 'fit') and callable(X.fit):
            # Don't get num_samples from an ensembles length!
            raise TypeError(message)

        if not hasattr(X, '__len__') and not hasattr(X, 'shape'):
            if hasattr(X, '__array__'):
                X = np.asarray(X)
            else:
                raise TypeError(message)

        if hasattr(X, 'shape') and X.shape is not None:
            if len(X.shape) == 0:
                raise TypeError("Singleton array %r cannot be considered"
                                " a valid collection." % X)
            # Check that shape is returning an integer or default to len
            # Dask dataframes may not return numeric shape[0] value
            if isinstance(X.shape[0], numbers.Integral):
                n_samples = X.shape[0]

        try:
            n_samples = len(X)
        except TypeError as type_error:
            raise TypeError(message) from type_error

        # continue #checksample_weight
        if dtype is not None and dtype not in [np.float32, np.float64]:
            dtype = np.float64

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=dtype)
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
        else:
            if dtype is None:
                dtype = [np.float64, np.float32]

            #CHECK ARRAY
            def check_array(array, accept_sparse=False, *, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, estimator=None):
                
                # store reference to original array to check if copy is needed when
                # function returns
                array_orig = array

                # store whether originally we wanted numeric dtype
                dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

                dtype_orig = getattr(array, "dtype", None)
                if not hasattr(dtype_orig, 'kind'):
                    # not a data type (e.g. a column named dtype in a pandas DataFrame)
                    dtype_orig = None

                # check if the object contains several dtypes (typically a pandas
                # DataFrame), and store them. If not, store None.
                dtypes_orig = list([])
                has_pd_integer_array = False
                if hasattr(array, "dtypes") and hasattr(array.dtypes, '__array__'):
                    # throw warning if columns are sparse. If all columns are sparse, then
                    # array.sparse exists and sparsity will be perserved (later).
                    with suppress(ImportError): dtypes_orig = list(array.dtypes)
                    # pandas boolean dtype __array__ interface coerces bools to objects
                    for i, dtype_iter in enumerate(dtypes_orig):
                        if dtype_iter.kind == 'b':
                            dtypes_orig[i] = np.dtype(object)
                        elif dtype_iter.name.startswith(("Int", "UInt")):
                            # name looks like an Integer Extension Array, now check for
                            # the dtype
                            with suppress(ImportError):
                                from pandas import (Int8Dtype, Int16Dtype,
                                                    Int32Dtype, Int64Dtype,
                                                    UInt8Dtype, UInt16Dtype,
                                                    UInt32Dtype, UInt64Dtype)
                                if isinstance(dtype_iter, (Int8Dtype, Int16Dtype,
                                                        Int32Dtype, Int64Dtype,
                                                        UInt8Dtype, UInt16Dtype,
                                                        UInt32Dtype, UInt64Dtype)):
                                    has_pd_integer_array = True

                    if all(isinstance(dtype, np.dtype) for dtype in dtypes_orig):
                        dtype_orig = np.result_type(*dtypes_orig)

                if dtype_numeric:
                    if dtype_orig is not None and dtype_orig.kind == "O":
                        # if input is object, convert to float.
                        dtype = np.float64
                    else:
                        dtype = None

                if isinstance(dtype, (list, tuple)):
                    if dtype_orig is not None and dtype_orig in dtype:
                        # no dtype conversion required
                        dtype = None
                    else:
                        # dtype conversion required. Let's select the first element of the
                        # list of accepted types
                        dtype = dtype[0]

                if has_pd_integer_array:
                    # If there are any pandas integer extension arrays,
                    array = array.astype(dtype)

                if force_all_finite not in (True, False, 'allow-nan'):
                    raise ValueError('force_all_finite should be a bool or "allow-nan"'
                                    '. Got {!r} instead'.format(force_all_finite))

                if estimator is not None:
                    if isinstance(estimator, str):
                        estimator_name = estimator
                    else:
                        estimator_name = estimator.__class__.__name__
                else:
                    estimator_name = "Estimator"
                context = " by %s" % estimator_name if estimator is not None else ""

                # When all dataframe columns are sparse, convert to a sparse array
                if hasattr(array, 'sparse') and array.ndim > 1:
                    # DataFrame.sparse only supports `to_coo`
                    array = array.sparse.to_coo()

                if sp.issparse(array):
                    _ensure_no_complex_data(array)
                    array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                                dtype=dtype, copy=copy,
                                                force_all_finite=force_all_finite,
                                                accept_large_sparse=accept_large_sparse)
                else:
                    # If np.array(..) gives ComplexWarning, then we convert the warning
                    # to an error. This is needed because specifying a non complex
                    # dtype to the function converts complex to real dtype,
                    # thereby passing the test made in the lines following the scope
                    # of warnings context manager.

                    # It is possible that the np.array(..) gave no warning. This happens
                    # when no dtype conversion happened, for example dtype = None. The
                    # result is that np.array(..) produces an array of complex dtype
                    # and we need to catch and raise exception for such cases.
                    _ensure_no_complex_data(array)

                    if ensure_2d:
                        # If input is scalar raise error
                        if array.ndim == 0:
                            raise ValueError(
                                "Expected 2D array, got scalar array instead:\narray={}.\n"
                                "Reshape your data either using array.reshape(-1, 1) if "
                                "your data has a single feature or array.reshape(1, -1) "
                                "if it contains a single sample.".format(array))
                        # If input is 1D raise error
                        if array.ndim == 1:
                            raise ValueError(
                                "Expected 2D array, got 1D array instead:\narray={}.\n"
                                "Reshape your data either using array.reshape(-1, 1) if "
                                "your data has a single feature or array.reshape(1, -1) "
                                "if it contains a single sample.".format(array))

                    # make sure we actually converted to numeric:
                    if dtype_numeric and array.dtype.kind == "O":
                        array = array.astype(np.float64)
                    if not allow_nd and array.ndim >= 3:
                        raise ValueError("Found array with dim %d. %s expected <= 2."
                                        % (array.ndim, estimator_name))

                    if force_all_finite:
                        _assert_all_finite(array,
                                        allow_nan=force_all_finite == 'allow-nan')

                if ensure_min_samples > 0:
                    n_samples = _num_samples(array)
                    if n_samples < ensure_min_samples:
                        raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                                        " minimum of %d is required%s."
                                        % (n_samples, array.shape, ensure_min_samples,
                                            context))

                if ensure_min_features > 0 and array.ndim == 2:
                    n_features = array.shape[1]
                    if n_features < ensure_min_features:
                        raise ValueError("Found array with %d feature(s) (shape=%s) while"
                                        " a minimum of %d is required%s."
                                        % (n_features, array.shape, ensure_min_features,
                                            context))

                if copy and np.may_share_memory(array, array_orig):
                    array = np.array(array, dtype=dtype, order=order)

                return array

            sample_weight = check_array(sample_weight, accept_sparse=False, ensure_2d=False, dtype=dtype, order="C")

            #continue #checksample_weight

            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight.shape == {}, expected {}!"
                                .format(sample_weight.shape, (n_samples,)))
        return sample_weight