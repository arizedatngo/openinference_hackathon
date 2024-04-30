from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPSpanExporterGRPC
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import os

def instrument():
    collector_endpoint = os.getenv("COLLECTOR_ENDPOINT")
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_exporter_HTTP = OTLPSpanExporterHTTP(endpoint=collector_endpoint)
    span_processor_HTTP = SimpleSpanProcessor(span_exporter=span_exporter_HTTP)
    tracer_provider.add_span_processor(span_processor=span_processor_HTTP)
    print("🔭 OpenInference instrumentation enabled.")

    # Import the automatic instrumentor from OpenInference
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    # Set the Space and API keys as headers for authentication
    headers = f"space_key={os.getenv('ARIZE_SPACE_KEY')},api_key={os.getenv('ARIZE_API_KEY')}"  # Retrieve API keys from environment
    os.environ['OTEL_EXPORTER_OTLP_TRACES_HEADERS'] = headers

    # Define the span processor as an exporter to the desired endpoint
    endpoint = "https://otlp.arize.com/v1"
    span_exporter_GRPC = OTLPSpanExporterGRPC(endpoint=endpoint)
    span_processor_GRPC = SimpleSpanProcessor(span_exporter=span_exporter_GRPC)

    # Set the tracer provider
    tracer_provider.add_span_processor(span_processor=span_processor_GRPC)
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument()
    print("🔭 OpenInference instrumentation enabled.")
