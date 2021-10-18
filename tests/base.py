from sample_stream import SampleStream
from darcyai_engine.output.output_stream import OutputStream
from darcyai_engine.tests.perceptor_mock import PerceptorMock
from darcyai_engine.pipeline import Pipeline

def perceptor_input_callback(input_data, pom):
    return input_data


input_stream = SampleStream(max_runs=2)
output_stream = OutputStream()

pipeline = Pipeline(input_stream)

pipeline.add_output_stream(output_stream)

p1 = PerceptorMock(model_path="models/p1.tflite")
pipeline.add_perceptor("p1", p1, accelerator_idx=0, input_callback=perceptor_input_callback)

pipeline.run()
