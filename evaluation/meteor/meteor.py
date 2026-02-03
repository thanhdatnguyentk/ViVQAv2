import os
import subprocess
import threading
import tarfile
from evaluation.utils import download_from_url

METEOR_GZ_URL = 'http://aimagelab.ing.unimore.it/speaksee/data/meteor.tgz'
METEOR_JAR = 'meteor-1.5.jar'

class Meteor:
    def __init__(self):
        # 1. Luôn khởi tạo Lock đầu tiên
        self.lock = threading.Lock()
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        jar_path = os.path.normpath(os.path.join(base_path, METEOR_JAR))
        gz_path = os.path.join(base_path, os.path.basename(METEOR_GZ_URL))
        
        if not os.path.isfile(jar_path):
            if not os.path.isfile(gz_path):
                download_from_url(METEOR_GZ_URL, gz_path)
            with tarfile.open(gz_path, "r") as tar:
                tar.extractall(path=base_path)
            os.remove(gz_path)

        # 2. Sử dụng danh sách cho cmd giúp subprocess tự xử lý các khoảng trắng trong path
        # Dùng '-l other' cho tiếng Việt
        self.meteor_cmd = ['java', '-Xmx512m', '-jar', jar_path, '-', '-', '-stdio', '-l', 'other']
        
        try:
            self.meteor_p = subprocess.Popen(
                self.meteor_cmd,
                cwd=base_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False, # Đổi thành False nếu đã dùng list (an toàn hơn)
                bufsize=0    # Binary mode không hỗ trợ line buffering
            )
            print('METEOR: Started.')
        except Exception as e:
            print(f"FAILED TO START METEOR: {e}")
            raise e

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = sorted(gts.keys())
        scores = []

        with self.lock: # Sử dụng 'with' để đảm bảo release lock tự động
            if self.meteor_p.poll() is not None:
                raise RuntimeError("METEOR process has terminated unexpectedly.")

            eval_line = 'EVAL'
            for i in imgIds:
                assert(len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += f' ||| {stat}'

            # Ghi dòng EVAL cuối cùng
            self._write_to_stdin(eval_line)
            
            for _ in range(len(imgIds)):
                line = self.meteor_p.stdout.readline().decode('utf-8').strip()
                scores.append(float(line))
            
            final_score_line = self.meteor_p.stdout.readline().decode('utf-8').strip()
            final_score = float(final_score_line)

        return final_score, scores

    def _stat(self, hypothesis_str, reference_list):
        # Làm sạch chuỗi triệt để
        def clean(s):
            return str(s).replace('|||', '').replace('\n', ' ').replace('\r', '').strip()

        hypothesis_str = clean(hypothesis_str)
        references = ' ||| '.join([clean(r) for r in reference_list])
        score_line = f'SCORE ||| {references} ||| {hypothesis_str}'
        
        self._write_to_stdin(score_line)
        
        raw = self.meteor_p.stdout.readline().decode('utf-8').strip()
        if not raw:
            raise RuntimeError("METEOR returned empty output.")
            
        numbers = [str(int(float(n))) for n in raw.split()]
        return ' '.join(numbers)

    def _write_to_stdin(self, line):
        """Hàm helper để ghi dữ liệu an toàn vào pipe trên Windows"""
        if self.meteor_p.poll() is not None:
            raise RuntimeError("METEOR process is not running.")
        
        # Đảm bảo kết thúc bằng \n và ép kiểu binary
        msg = f"{line}\n".encode('utf-8')
        try:
            self.meteor_p.stdin.write(msg)
            self.meteor_p.stdin.flush()
        except OSError as e:
            if e.errno == 22:
                # Log thêm thông tin nếu gặp lỗi 22
                print(f"DEBUG: Failed to write to pipe. Msg length: {len(msg)}")
            raise e

    def __del__(self):
        # Đóng subprocess một cách an toàn
        try:
            if hasattr(self, 'meteor_p') and self.meteor_p:
                if self.meteor_p.stdin:
                    self.meteor_p.stdin.close()
                self.meteor_p.kill()
                self.meteor_p.wait()
        except:
            pass

    def __str__(self):
        return 'METEOR'