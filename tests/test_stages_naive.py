from llm_serving.stages.base import STAGES
from llm_serving.stages.naive import TimestampStreamer


def test_timestamp_streamer_skips_the_first_put_call():
    # generate() calls put() once with the full prompt before generation starts.
    streamer = TimestampStreamer()
    streamer.put("prompt_echo")

    assert streamer.timestamps == []


def test_timestamp_streamer_records_one_timestamp_per_put_call_after_the_first():
    streamer = TimestampStreamer()
    streamer.put("prompt_echo")
    streamer.put("token_a")
    streamer.put("token_b")
    streamer.put("token_c")

    assert len(streamer.timestamps) == 3


def test_timestamp_streamer_timestamps_are_increasing():
    streamer = TimestampStreamer()
    streamer.put("prompt_echo")
    streamer.put("a")
    streamer.put("b")

    assert streamer.timestamps[1] >= streamer.timestamps[0]


def test_naive_stage_registers_itself():
    assert "naive" in STAGES
