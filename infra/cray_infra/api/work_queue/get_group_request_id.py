def get_group_request_id(request_id):
    # The id is the part before the first underscore
    return request_id.split("_")[0]
