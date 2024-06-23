import subprocess
import time
import threading

class ProcessController:

    def __init__(self, verbose=1, commands=[], flags=[]):
        self.verbose = verbose
        
        self.commands = ['cargo', 'run']
        if commands:
            self.commands = commands
            
        self.flags = ['--release']
        if flags:
            self.flags = flags
            
        self.process = None

        if verbose == 0:
            self.stdout = subprocess.DEVNULL
            self.stderr = subprocess.DEVNULL
        if verbose == 1:
            self.stdout = subprocess.DEVNULL
            self.stderr = subprocess.PIPE
        elif verbose >= 2:
            self.stdout = subprocess.PIPE
            self.stderr = subprocess.PIPE


    def start_simulation(self):
        for flag in self.flags:
            if flag not in self.commands:
                self.commands.append(flag)

        print(f"Running: \n\t{' '.join(self.commands)}")
        self.process = subprocess.Popen(
            self.commands,
            stdout=self.stdout,
            stderr=self.stderr,
            text=True,
            bufsize=1
        )

        if self.verbose >= 1:
            threading.Thread(target=self._read_output, args=(self.process.stdout, "STDOUT")).start()
        if self.verbose >= 2:
            threading.Thread(target=self._read_output, args=(self.process.stderr, "STDERR")).start()

        threading.Thread(target=self._monitor_simulation).start()


    def _read_output(self, pipe, pipe_name):
        if pipe:
            for line in iter(pipe.readline, ''):
                if line:
                    print(f"[{pipe_name}] {line}", end='')


    def _monitor_simulation(self):
        try:
            while True:
                return_code = self.process.poll()
                if return_code is not None:
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            print("Process interrupted by user")
            self.terminate_simulation()
        
        self.process.wait()
        print("Process terminated")

        
    def terminate_simulation(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Process terminated")
