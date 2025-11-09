
Author: Lee Boonstra 


"You don’t need to be a data scientist or a machine learning  engineer – everyone can write  a prompt."

Node: [[books]]
Tags: #book , #ai
## Introduce

Hãy nhớ cách một mô hình ngôn ngữ lớn (LLM) hoạt động; nó là một **công cụ dự đoán**. Mô hình nhận **văn bản tuần tự** làm đầu vào, rồi **dự đoán token (đơn vị ngôn ngữ) tiếp theo** sẽ là gì, dựa trên dữ liệu mà nó đã được huấn luyện.

LLM được vận hành để lặp lại quá trình này nhiều lần — mỗi lần, nó **thêm token vừa được dự đoán** vào cuối chuỗi văn bản hiện có để tiếp tục **dự đoán token kế tiếp**.

Việc dự đoán token tiếp theo **dựa trên mối quan hệ** giữa các token trước đó và **những gì LLM đã học được trong quá trình huấn luyện**.

Khi bạn viết một **prompt** (lời nhắc), bạn đang cố gắng thiết lập cho **LLM** (mô hình ngôn ngữ lớn) dự đoán đúng chuỗi **token** (đơn vị ngôn ngữ) tiếp theo. **Kỹ thuật thiết kế prompt** (prompt engineering) là quá trình tạo ra những prompt chất lượng cao nhằm hướng dẫn LLM tạo ra các kết quả chính xác.

Quá trình này bao gồm việc thử nghiệm để tìm ra prompt hiệu quả nhất, tối ưu độ dài của prompt, và đánh giá phong cách viết cũng như cấu trúc của prompt trong mối liên hệ với nhiệm vụ được giao.

Trong bối cảnh **xử lý ngôn ngữ tự nhiên (NLP)** và **LLM**, một _prompt_ là phần đầu vào được cung cấp cho mô hình để nó tạo ra phản hồi hoặc dự đoán.

Các lời nhắc (prompts) này có thể được sử dụng để thực hiện nhiều loại nhiệm vụ hiểu và tạo nội dung khác nhau, chẳng hạn như **tóm tắt văn bản, trích xuất thông tin, hỏi đáp, phân loại văn bản, dịch ngôn ngữ hoặc mã, tạo mã, và viết tài liệu hoặc suy luận về mã**.

Bạn có thể tham khảo **các hướng dẫn về cách tạo prompt của Google** với những ví dụ đơn giản và hiệu quả.

Khi thực hiện **kỹ thuật thiết kế prompt (prompt engineering)**, bạn sẽ bắt đầu bằng việc chọn một mô hình. Các prompt có thể cần được **tối ưu hóa cho mô hình cụ thể** mà bạn sử dụng, dù đó là **mô hình ngôn ngữ Gemini trong Vertex AI, GPT, Claude**, hay **một mô hình mã nguồn mở như Gemma hoặc LLaMA**.

Ngoài phần prompt, bạn cũng sẽ cần **điều chỉnh nhiều cấu hình khác nhau của mô hình ngôn ngữ lớn (LLM)**.

## LLM output configuration

Khi bạn đã chọn được mô hình của mình, bạn sẽ cần xác định cấu hình của mô hình đó. Hầu hết các mô hình ngôn ngữ lớn (LLM) đều có nhiều tùy chọn cấu hình khác nhau để kiểm soát đầu ra của chúng. Việc thiết kế prompt (prompt engineering) hiệu quả đòi hỏi phải thiết lập các cấu hình này một cách tối ưu cho nhiệm vụ của bạn.

### Output length

Một thiết lập cấu hình quan trọng là số lượng **token** được tạo ra trong một phản hồi. Việc tạo ra nhiều **token** hơn đòi hỏi mô hình ngôn ngữ lớn (LLM) phải thực hiện nhiều phép tính hơn, dẫn đến **mức tiêu thụ năng lượng cao hơn**, **thời gian phản hồi có thể chậm hơn**, và **chi phí cao hơn**.

Việc giảm độ dài đầu ra của mô hình ngôn ngữ lớn (LLM) **không khiến LLM trở nên ngắn gọn hơn về mặt phong cách hay nội dung** trong văn bản mà nó tạo ra — điều đó chỉ khiến LLM **ngừng sinh thêm token** khi đạt đến giới hạn được đặt ra.

Nếu bạn cần một đầu ra ngắn, bạn có thể sẽ phải **thiết kế lại prompt** của mình để phù hợp với yêu cầu đó.

Giới hạn độ dài đầu ra đặc biệt quan trọng trong một số kỹ thuật gợi ý (prompting) của LLM, như **ReAct**, nơi mà mô hình có thể tiếp tục sinh ra các token không cần thiết sau khi đã tạo xong phản hồi mong muốn.

Hãy lưu ý rằng, **việc tạo ra nhiều token hơn** sẽ **đòi hỏi nhiều tài nguyên tính toán hơn** từ LLM, dẫn đến **tiêu thụ năng lượng cao hơn** và **thời gian phản hồi chậm hơn**, từ đó **làm tăng chi phí**.

### Sampling controls

Các mô hình ngôn ngữ lớn (LLM) không thực sự “dự đoán” một token duy nhất. Thay vào đó, LLM dự đoán **xác suất** cho từng khả năng về token tiếp theo — mỗi token trong **từ vựng** của mô hình sẽ được gán một xác suất. Sau đó, các xác suất này được **lấy mẫu (sampling)** để xác định token nào sẽ được tạo ra tiếp theo.

Ba tham số cấu hình phổ biến nhất — **temperature**, **top-K**, và **top-P** — quyết định cách các xác suất dự đoán được xử lý để chọn ra **một token đầu ra duy nhất**.

### Temperature

Nhiệt độ kiểm soát mức độ ngẫu nhiên trong việc chọn token. Nhiệt độ thấp phù hợp cho các yêu cầu (prompt) cần phản hồi có tính xác định cao, trong khi nhiệt độ cao có thể dẫn đến kết quả đa dạng hoặc bất ngờ hơn.

Nhiệt độ bằng 0 (giải mã tham lam — _greedy decoding_) là xác định: token có xác suất cao nhất luôn được chọn (tuy nhiên, nếu có hai token có cùng xác suất cao nhất, tùy vào cách hệ thống xử lý tình huống “hòa”, bạn có thể không nhận được cùng một đầu ra ngay cả khi nhiệt độ là 0).

Khi nhiệt độ tiến gần giá trị tối đa, đầu ra có xu hướng ngẫu nhiên hơn. Và khi nhiệt độ càng tăng cao, mọi token đều có khả năng như nhau để trở thành token được dự đoán tiếp theo.

Cơ chế điều khiển nhiệt độ của Gemini có thể được hiểu tương tự như hàm _softmax_ được sử dụng trong học máy. Cài đặt nhiệt độ thấp tương ứng với nhiệt độ _softmax_ thấp (T), nhấn mạnh một lựa chọn đơn lẻ, ưu tiên với độ chắc chắn cao. Ngược lại, cài đặt nhiệt độ Gemini cao giống với nhiệt độ _softmax_ cao, khiến một phạm vi rộng hơn của các giá trị nhiệt độ xung quanh mức đã chọn trở nên chấp nhận được hơn.

Sự không chắc chắn tăng lên này phù hợp cho các tình huống mà độ chính xác cứng nhắc không quá cần thiết — chẳng hạn như khi thử nghiệm với các đầu ra mang tính sáng tạo.


### Top-K and top-P

Top-K và top-P (còn được gọi là **lấy mẫu hạt nhân**) là hai thiết lập lấy mẫu được sử dụng trong các mô hình ngôn ngữ lớn (LLM) nhằm giới hạn việc chọn **token** tiếp theo trong số những token có **xác suất dự đoán cao nhất**. Giống như **nhiệt độ (temperature)**, các thiết lập lấy mẫu này điều chỉnh **mức độ ngẫu nhiên và đa dạng** của văn bản được tạo ra.

- Phép lấy mẫu Top-K chọn ra **K token có xác suất cao nhất** từ phân phối dự đoán của mô hình. Giá trị **K càng cao**, đầu ra của mô hình càng **sáng tạo và đa dạng**;  giá trị **K càng thấp**, đầu ra càng **ổn định và mang tính chính xác thực tế** hơn. Khi **top-K = 1**, quá trình này tương đương với **giải mã tham lam (greedy decoding)**.
- Lấy mẫu Top-P (Top-P sampling) chọn ra các token hàng đầu sao cho tổng xác suất tích lũy của chúng không vượt quá một giá trị nhất định (P). Giá trị của P dao động từ 0 (giải mã tham lam — greedy decoding) đến 1 (bao gồm tất cả các token trong vốn từ vựng của mô hình ngôn ngữ lớn).

Cách tốt nhất để lựa chọn giữa **top-K** và **top-P** là **thử nghiệm với cả hai phương pháp** (hoặc kết hợp cả hai) và **xem phương pháp nào tạo ra kết quả mà bạn mong muốn**.


### Putting it all together 


