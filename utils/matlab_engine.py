import matlab.engine
eng = matlab.engine.start_matlab()

[uni_label_of_from_atalas, max_prop, matching_idx] = eng.lc_mapping_brain_atalas_highlevel(nargout=3)

print(matching_idx)