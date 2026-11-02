import os
import time
import google.genai as genai
from google.genai import types
import gradio as gr
from PIL import Image

# เริ่มจากโมเดลที่เร็วที่สุดก่อน; ตั้ง GEMINI_MODEL ใน Render เพื่อเลือกตัวแรกเองได้
_env_model = os.getenv("GEMINI_MODEL", "").strip()
MODELS = list(dict.fromkeys(
    ([_env_model] if _env_model else []) + ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
))

# ถ้าโมเดลไหนไม่ตอบภายในเวลานี้ (วินาที) จะข้ามไปโมเดลถัดไป
TIMEOUT_SEC = int(os.getenv("GEMINI_TIMEOUT", "45"))


def make_config(thinking=True):
    base = dict(
        temperature=0.1,
        max_output_tokens=1500,  # จำกัดความยาวคำตอบ ไม่ให้ตอบยาวเกินจำเป็น
        http_options=types.HttpOptions(timeout=TIMEOUT_SEC * 1000),
    )
    if thinking:
        try:
            return types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="minimal"), **base
            )
        except Exception:
            pass
    return types.GenerateContentConfig(**base)


def check_object(image, expected_list, api_key):

    if not api_key:
        yield "กรุณากรอก Gemini API Key ก่อนค่ะ"
        return

    if image is None:
        yield "กรุณาอัปโหลดภาพก่อนค่ะ"
        return

    if not expected_list:
        yield "กรุณาเลือกอุปกรณ์ PPE อย่างน้อย 1 รายการค่ะ"
        return

    try:
        img = image.convert("RGB")
        # ย่อภาพก่อนส่ง: ภาพจากมือถือมักใหญ่หลาย MB ทำให้อัปโหลดและประมวลผลช้า
        img.thumbnail((768, 768))

        client = genai.Client(api_key=api_key.strip())

        items_to_check = ", ".join(expected_list)

        prompt = f"""
คุณเป็น AI ผู้ช่วยตรวจสอบอุปกรณ์ป้องกันส่วนบุคคล (PPE)
สำหรับห้องปฏิบัติการเคมี

กรุณาวิเคราะห์ภาพที่แนบมาอย่างละเอียด
โดยตรวจสอบอุปกรณ์ต่อไปนี้: {items_to_check}

ข้อกำหนดในการตรวจสอบ:

1. ถุงมือ
- ตรวจสอบถุงมือทุกสี เช่น สีขาว น้ำเงิน ม่วง
  เหลือง ดำ เขียว ชมพู และสีอื่น ๆ
- ตรวจสอบทั้งถุงมือยางและถุงมือไนไตรล์
- ห้ามตัดสินว่าถุงมือไม่พบเพียงเพราะสีแตกต่างจากสีขาว

2. แว่นตานิรภัย
- ตรวจสอบแว่นครอบตาและแว่นตานิรภัย
- พิจารณาจากรูปทรง เลนส์ และลักษณะของแว่น

3. เสื้อช็อป
- ตรวจสอบเสื้อช็อปวิศวกรรม เสื้อเครื่องแบบช่าง
  และเสื้อปฏิบัติงาน
- พิจารณาจากรูปแบบเสื้อและลักษณะเครื่องแต่งกาย
- หากไม่สามารถยืนยันได้ว่าเป็นชุดช็อปจริง
  ให้ระบุว่าไม่สามารถยืนยันได้

4. สายสะท้อนแสง
- ตรวจสอบแถบสะท้อนแสงบนเสื้อหรือเครื่องแต่งกาย
- ไม่จำกัดสี เช่น สีเงิน สีเหลือง หรือสีส้ม

5. หน้ากากอนามัย
- ตรวจสอบหน้ากากที่สวมปิดจมูกและปาก
- ตรวจสอบทุกสี เช่น สีขาว ฟ้า เขียว ดำ และสีอื่น ๆ

หลักการสำคัญ:
- ตรวจสอบจากรูปร่าง ลักษณะ และหน้าที่ของอุปกรณ์
  ไม่ใช้สีเป็นเงื่อนไขตัดสินเพียงอย่างเดียว
- ตรวจสอบทุกประเภทที่ผู้ใช้เลือก
- หากพบหลายประเภท ให้รายงานทุกประเภท
- ห้ามแต่งข้อมูลหรือคาดเดาว่ามีอุปกรณ์
  หากมองไม่เห็นชัดเจน
- หากภาพไม่ชัด ให้ระบุว่าไม่สามารถยืนยันได้

กรุณาตอบเป็นภาษาไทยในรูปแบบดังนี้:

ผลการตรวจสอบ PPE

- ชื่ออุปกรณ์: พบ / ไม่พบ / ไม่สามารถยืนยันได้
- รายละเอียด: อธิบายสี ลักษณะ และสิ่งที่สังเกตเห็น

สรุปผลการตรวจสอบ:
อธิบายว่าอุปกรณ์ใดพบและอุปกรณ์ใดไม่พบ

หมายเหตุ:
การวิเคราะห์จากภาพเป็นเพียงการประเมินเบื้องต้น
ไม่ใช่การรับรองความปลอดภัยในการเข้าห้องปฏิบัติการ

ตอบให้กระชับ รายละเอียดของแต่ละอุปกรณ์ไม่เกิน 1-2 ประโยค
"""

        last_error = ""
        for model in MODELS:
            use_thinking = True
            for attempt in range(2):
                text = ""
                try:
                    stream = client.models.generate_content_stream(
                        model=model,
                        contents=[img, prompt],
                        config=make_config(use_thinking),
                    )
                    for chunk in stream:
                        if chunk.text:
                            text += chunk.text
                            yield text  # แสดงผลทีละส่วนระหว่างที่ AI กำลังเขียน
                    if text.strip():
                        return
                    last_error = "AI ไม่ส่งผลการวิเคราะห์กลับมา"
                    break

                except Exception as e:
                    error = str(e)
                    last_error = error
                    low = error.lower()

                    # ได้ผลมาเกือบครบแล้วแต่หลุดกลางทาง -> แสดงเท่าที่ได้
                    if len(text) > 200:
                        yield text + "\n\n_⚠️ การเชื่อมต่อขาดก่อนผลจะครบ กดตรวจสอบอีกครั้งหากต้องการผลเต็ม_"
                        return

                    if "400" in error and "thinking" in low and use_thinking:
                        use_thinking = False
                        continue

                    if "401" in error or "403" in error or "API_KEY_INVALID" in error or "PERMISSION_DENIED" in error:
                        yield "API Key ไม่ถูกต้องหรือไม่มีสิทธิ์ใช้งาน กรุณาตรวจสอบ API Key ใน Google AI Studio"
                        return

                    if "503" in error or "UNAVAILABLE" in error:
                        if attempt == 0:
                            time.sleep(1)
                            continue
                        break

                    if ("504" in error or "DEADLINE_EXCEEDED" in error or "timeout" in low
                            or "timed out" in low or "404" in error or "NOT_FOUND" in error
                            or "429" in error or "RESOURCE_EXHAUSTED" in error):
                        break  # ช้าเกิน/ไม่พบ/โควตาหมด -> ลองโมเดลถัดไปทันที

                    yield f"เกิดข้อผิดพลาดในการประมวลผลภาพ ({model}): {error}"
                    return

            yield f"⏳ โมเดล {model} ตอบช้า กำลังลองโมเดลถัดไป..."

        if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
            yield "ใช้งาน API เกินโควตา กรุณารอสักครู่แล้วลองใหม่อีกครั้ง"
        else:
            yield f"ระบบ Gemini ตอบช้าหรือไม่ว่างในขณะนี้ กรุณาลองใหม่อีกครั้งในอีกสักครู่\n\n(รายละเอียด: {last_error})"

    except Exception as e:
        yield f"เกิดข้อผิดพลาด: {str(e)}"


PPE_CHOICES = ["ถุงมือ", "แว่นตานิรภัย", "เสื้อช็อป", "สายสะท้อนแสง", "หน้ากากอนามัย"]

THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="cyan",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Prompt"), "sans-serif"],
)

CSS = """
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
#hero {
    background: linear-gradient(135deg, #0f766e 0%, #0e7490 100%);
    border-radius: 18px; padding: 26px 30px; margin-bottom: 8px;
}
#hero h1 { color: #fff !important; margin: 0 0 4px 0; font-size: 2rem; }
#hero p { color: #ccfbf1 !important; margin: 0; font-size: 1.05rem; }
.panel {
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px !important; padding: 20px !important;
    background: #ffffff !important;
    box-shadow: 0 6px 24px rgba(15, 23, 42, 0.07) !important;
}
.dark .panel { background: #0f172a !important; border-color: #1e293b !important; }
.panel-title h3 { margin: 0 0 8px 0 !important; font-size: 1.15rem !important; }
#btn-row button { min-height: 52px !important; font-size: 1.05rem !important; }
#result-box {
    min-height: 560px; line-height: 1.8; font-size: 1rem;
    background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 14px; padding: 18px 20px !important;
}
.dark #result-box { background: #111827; border-color: #334155; }
#footer p { text-align: center; color: var(--body-text-color-subdued) !important; font-size: 0.85rem; }
"""

# Gradio 6 รับ theme/css ที่ launch(); Gradio 5 รับที่ gr.Blocks()
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6
STYLE = dict(theme=THEME, css=CSS)

with gr.Blocks(title="LabSafe AI", **({} if GRADIO_6 else STYLE)) as demo:

    gr.HTML(
        "<h1>🧪 LabSafe AI</h1>"
        "<p>ระบบช่วยตรวจสอบอุปกรณ์ป้องกันส่วนบุคคล (PPE) จากภาพอัตโนมัติ สำหรับห้องปฏิบัติการเคมี</p>",
        elem_id="hero",
    )

    with gr.Row(equal_height=False):

        # ---------- ซ้าย: กรอกข้อมูล ----------
        with gr.Column(scale=5, elem_classes="panel"):
            gr.Markdown("### 📝 ข้อมูลสำหรับตรวจสอบ", elem_classes="panel-title")

            api_key_input = gr.Textbox(
                label="🔑 Gemini API Key",
                type="password",
                placeholder="วาง API Key ที่นี่"
            )

            expected = gr.CheckboxGroup(
                choices=PPE_CHOICES,
                value=PPE_CHOICES,
                label="🦺 เลือกประเภทวัตถุที่ต้องการตรวจสอบ"
            )

            image = gr.Image(
                type="pil",
                label="📷 ถ่ายรูปหรืออัปโหลดภาพ",
                height=340
            )

            with gr.Row(elem_id="btn-row"):
                clear_btn = gr.Button("ล้างข้อมูล", variant="secondary", scale=1)
                button = gr.Button("🔍 ตรวจสอบ PPE", variant="primary", scale=2, elem_id="check-btn")

        # ---------- ขวา: ผลการตรวจสอบ ----------
        with gr.Column(scale=6, elem_classes="panel"):
            gr.Markdown("### 📋 ผลการตรวจสอบ", elem_classes="panel-title")
            result = gr.Markdown(
                "_ผลการวิเคราะห์จะแสดงที่นี่ หลังจากกดปุ่ม **ตรวจสอบ PPE**_",
                elem_id="result-box"
            )

    gr.Markdown(
        "⚠️ การวิเคราะห์จากภาพเป็นเพียงการประเมินเบื้องต้น ไม่ใช่การรับรองความปลอดภัยในการเข้าห้องปฏิบัติการ",
        elem_id="footer"
    )

    button.click(
        fn=lambda: "⏳ กำลังวิเคราะห์ภาพ กรุณารอสักครู่...",
        outputs=result,
        queue=False,
    ).then(
        fn=check_object,
        inputs=[image, expected, api_key_input],
        outputs=result
    )

    clear_btn.click(
        fn=lambda: (None, PPE_CHOICES, "_ผลการวิเคราะห์จะแสดงที่นี่ หลังจากกดปุ่ม **ตรวจสอบ PPE**_"),
        outputs=[image, expected, result],
        queue=False,
    )

if __name__ == "__main__":
    # Render ต้องการให้แอปฟังที่ 0.0.0.0 และพอร์ตจากตัวแปร PORT
    launch_kwargs = dict(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
    if GRADIO_6:
        launch_kwargs.update(STYLE)
    demo.queue(default_concurrency_limit=4).launch(**launch_kwargs)
