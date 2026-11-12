"""A plain continuation has to receive loss even when it has no protocol token."""
import importlib.util
from pathlib import Path
import unittest

script = Path(__file__).resolve().parents[1] / 'SFT-Scripts' / 'audit_encoded_lengths.py'
spec = importlib.util.spec_from_file_location('second_encoded_audit', script)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ChatTokenizer:
    def encode(self, text, add_special_tokens):
        assert text == '<|im_start|>assistant\n' and not add_special_tokens
        return [1, 2, 3]

    def convert_tokens_to_ids(self, text):
        assert text == '<|im_end|>'
        return 4

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        words = {5: '<zh>', 6: '<spk1>', 7: '请联系', 8: '<|WAIT|>', 9: '张伟', 10: '。'}
        return ''.join(words[token] for token in ids)


class EncodedTargetsTest(unittest.TestCase):
    def setUp(self):
        texts = ['<zh><spk1>请联系', '<|WAIT|>', '张伟。']
        self.messages = [{'role': 'assistant', 'content': text} for text in texts]
        self.encoded = {'input_ids': [], 'labels': []}
        for payload in ([5, 6, 7], [8], [9, 10]):
            self.encoded['input_ids'].extend([1, 2, 3, *payload, 4])
            self.encoded['labels'].extend([-100, -100, -100, *payload, 4])

    def test_wait_and_plain_continuation_are_supervised(self):
        audit.check_assistant_turns(self.messages, self.encoded, ChatTokenizer())

    def test_masking_plain_text_is_detected(self):
        self.encoded['labels'][-3] = -100
        with self.assertRaisesRegex(ValueError, 'turn 3'):
            audit.check_assistant_turns(self.messages, self.encoded, ChatTokenizer())

    def test_altered_plain_text_is_detected(self):
        self.messages[2]['content'] = '张伟！'
        with self.assertRaisesRegex(ValueError, 'turn 3'):
            audit.check_assistant_turns(self.messages, self.encoded, ChatTokenizer())
