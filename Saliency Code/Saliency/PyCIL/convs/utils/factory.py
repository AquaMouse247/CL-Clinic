def get_model(model_name, args):
    name = model_name.lower()
    if name == "der":
        from Saliency.PyCIL.models.der import DER
        return DER(args)
    elif name == "foster":
        from Saliency.PyCIL.models.foster import FOSTER
        return FOSTER(args)
    elif name == "memo":
        from Saliency.PyCIL.models.memo import MEMO
        return MEMO(args)
    else:
        assert 0
