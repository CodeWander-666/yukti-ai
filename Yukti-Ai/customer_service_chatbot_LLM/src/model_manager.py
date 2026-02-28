20:43:11] 📦 Processing dependencies...

[20:43:11] 📦 Processed dependencies!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

[20:43:12] 🔄 Updated app!

2026-02-28 20:43:13.176 503 GET /script-health-check (127.0.0.1) 1057.74ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:43:18.176 503 GET /script-health-check (127.0.0.1) 1053.29ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:43:23.164 503 GET /script-health-check (127.0.0.1) 1039.57ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:43:28.185 503 GET /script-health-check (127.0.0.1) 1064.27ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:43:33.168 503 GET /script-health-check (127.0.0.1) 1048.24ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:43:38.157 503 GET /script-health-check (127.0.0.1) 1036.82ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:43:43.149 503 GET /script-health-check (127.0.0.1) 1033.74ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:43:48.150 503 GET /script-health-check (127.0.0.1) 1032.65ms

[20:43:51] 🐙 Pulling code changes from Github...

[20:43:51] 📦 Processing dependencies...

[20:43:51] 📦 Processed dependencies!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:43:53.203 503 GET /script-health-check (127.0.0.1) 1054.87ms

[20:43:53] 🔄 Updated app!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:43:58.209 503 GET /script-health-check (127.0.0.1) 1062.69ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:03.129 503 GET /script-health-check (127.0.0.1) 999.62ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:08.209 503 GET /script-health-check (127.0.0.1) 1081.41ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:24 in   

  <module>                                                                      

                                                                                

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

  ❱  24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

     26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/think.py:11 in  

  <module>                                                                      

                                                                                

      8 from typing import List, Dict, Any                                      

      9                                                                         

     10 # Import model manager (which now uses LLM package)                     

  ❱  11 from model_manager import load_model, get_model_config                  

     12 from langchain_helper import retrieve_documents  # unchanged            

     13                                                                         

     14 logger = logging.getLogger(__name__)                                    

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/model_manager.  

  py:7 in <module>                                                              

                                                                                

     4 """                                                                      

     5                                                                          

     6 import logging                                                           

  ❱  7 from LLM.zhipu import ZhipuSyncClient, TaskQueue, MODELS, get_model_con  

     8 from pathlib import Path                                                 

     9                                                                          

    10 logger = logging.getLogger(__name__)                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:13.153 503 GET /script-health-check (127.0.0.1) 1030.74ms

[20:44:16] 🐙 Pulling code changes from Github...

[20:44:17] 📦 Processing dependencies...

[20:44:17] 📦 Processed dependencies!

Zhipu import failed: No module named 'LLM'. Falling back to Gemini.

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:18.258 503 GET /script-health-check (127.0.0.1) 1126.85ms

[20:44:18] 🔄 Updated app!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:23.133 503 GET /script-health-check (127.0.0.1) 1008.05ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:28.159 503 GET /script-health-check (127.0.0.1) 1036.18ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:33.140 503 GET /script-health-check (127.0.0.1) 1018.79ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:38.202 503 GET /script-health-check (127.0.0.1) 1079.45ms

[20:44:40] 🐙 Pulling code changes from Github...

[20:44:41] 📦 Processing dependencies...

[20:44:41] 📦 Processed dependencies!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

[20:44:42] 🔄 Updated app!

2026-02-28 20:44:43.196 503 GET /script-health-check (127.0.0.1) 1069.45ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:48.208 503 GET /script-health-check (127.0.0.1) 1087.38ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:53.146 503 GET /script-health-check (127.0.0.1) 1018.36ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:44:58.199 503 GET /script-health-check (127.0.0.1) 1062.75ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:03.175 503 GET /script-health-check (127.0.0.1) 1051.52ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:08.197 503 GET /script-health-check (127.0.0.1) 1075.14ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:26 in   

  <module>                                                                      

                                                                                

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     24 from think import think                                                 

     25 from model_manager import get_available_models, MODELS                  

  ❱  26 from LLM.zhipu.queue_manager import TaskQueue                           

     27                                                                         

     28 # ---------- Page Configuration ----------                              

     29 st.set_page_config(                                                     

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:13.151 503 GET /script-health-check (127.0.0.1) 1024.28ms

[20:45:14] 🐙 Pulling code changes from Github...

Zhipu import failed: No module named 'LLM'. Falling back to Gemini.

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 import streamlit as st                                                  

     18 import pandas as pd                                                     

     19 from langchain_core.documents import Document                           

  ❱  20 from langchain_community.vectorstores import FAISS                      

     21                                                                         

     22 # Local imports                                                         

     23 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

[20:45:14] 📦 Processing dependencies...

[20:45:14] 📦 Processed dependencies!

[20:45:16] 🔄 Updated app!

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:18.142 503 GET /script-health-check (127.0.0.1) 1020.81ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:23.117 503 GET /script-health-check (127.0.0.1) 996.12ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:28.188 503 GET /script-health-check (127.0.0.1) 1067.98ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:33.115 503 GET /script-health-check (127.0.0.1) 992.54ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:38.161 503 GET /script-health-check (127.0.0.1) 1036.39ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:43.122 503 GET /script-health-check (127.0.0.1) 998.71ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:48.182 503 GET /script-health-check (127.0.0.1) 1054.77ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:53.130 503 GET /script-health-check (127.0.0.1) 1003.86ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:45:58.192 503 GET /script-health-check (127.0.0.1) 1048.18ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:03.130 503 GET /script-health-check (127.0.0.1) 1003.76ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:08.159 503 GET /script-health-check (127.0.0.1) 1041.70ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:13.151 503 GET /script-health-check (127.0.0.1) 1025.96ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:18.152 503 GET /script-health-check (127.0.0.1) 1027.40ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:23.171 503 GET /script-health-check (127.0.0.1) 1021.33ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:28.206 503 GET /script-health-check (127.0.0.1) 1074.40ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:33.165 503 GET /script-health-check (127.0.0.1) 1026.16ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:38.195 503 GET /script-health-check (127.0.0.1) 1072.24ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:43.216 503 GET /script-health-check (127.0.0.1) 1078.34ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:48.229 503 GET /script-health-check (127.0.0.1) 1092.70ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:53.222 503 GET /script-health-check (127.0.0.1) 1099.60ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:46:58.206 503 GET /script-health-check (127.0.0.1) 1068.89ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:47:03.225 503 GET /script-health-check (127.0.0.1) 1095.84ms

────────────────────── Traceback (most recent call last) ───────────────────────

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/exec_code.py:129 in exec_func_with_error_handling                        

                                                                                

  /home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/scriptru  

  nner/script_runner.py:689 in code_to_exec                                     

                                                                                

  /mount/src/yukti-ai/Yukti-Ai/customer_service_chatbot_LLM/src/main.py:20 in   

  <module>                                                                      

                                                                                

     17 from langchain_helper import get_embeddings, VECTORDB_PATH, BASE_DIR,   

     18 from think import think                                                 

     19 from model_manager import get_available_models, MODELS                  

  ❱  20 from LLM.zhipu.queue_manager import TaskQueue                           

     21                                                                         

     22 # Page config                                                           

     23 st.set_page_config(page_title="Yukti AI", page_icon="✨", layout="wide  

────────────────────────────────────────────────────────────────────────────────

ModuleNotFoundError: No module named 'LLM'

2026-02-28 20:47:08.168 503 GET /script-health-check (127.0.0.1) 1041.83ms

[20:47:10] 🐙 Pulling code changes from Github...

Zhipu import failed: No module named 'LLM'. Falling back to Gemini.

[20:47:11] 📦 Processing dependencies...

[20:47:11] 📦 Processed dependencies!

[20:47:12] 🔄 Updated app!
