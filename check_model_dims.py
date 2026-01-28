import onnxruntime as ort

# Load the student model
policy_path = "logs/MotionTracking/phuma_student/exported/model_15000.onnx"
policy = ort.InferenceSession(policy_path)

print("Model Inputs:")
for inp in policy.get_inputs():
    print("  {}: {}".format(inp.name, inp.shape))

print("\nModel Outputs:")
for out in policy.get_outputs():
    print("  {}: {}".format(out.name, out.shape))
