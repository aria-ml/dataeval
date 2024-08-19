from typing import Dict


def _validate_dict(d: Dict) -> None:
    """
    Verify that dict-of-arrays (proxy for dataframe) contains arrays of equal
    length.  Future iterations could include type checking, conversion from
    string to numeric types, etc.

    Parameters
    ----------
    d: Dict
        dictionary of {variable_name: values}
    """
    # assert that length of all arrays are equal -- could expand to other properties
    lengths = []
    for arr in d.values():
        lengths.append(arr.shape)

    if lengths[1:] != lengths[:-1]:
        raise ValueError("The lengths of each entry in the dictionary are not equal." f" Found lengths {lengths}")
