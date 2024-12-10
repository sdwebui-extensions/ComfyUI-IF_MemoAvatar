import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Create base styles for buttons
const style = document.createElement('style');
style.textContent = `
    .if-button {
        background: var(--comfy-input-bg);
        border: 1px solid var(--border-color);
        color: var(--input-text);
        padding: 4px 12px;
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-right: 5px;
    }
    
    .if-button:hover {
        background: var(--comfy-input-bg-hover);
    }
`;
document.head.appendChild(style);

app.registerExtension({
    name: "Comfy.IF_MemoAvatar",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IF_MemoAvatar") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function() {
            const result = origOnNodeCreated?.apply(this, arguments);

            // Find the audio widget
            const audioWidget = this.widgets?.find(w => w.name === "audio");
            if (audioWidget) {
                // Store original methods
                const origSetValue = audioWidget.setValue;
                const origCallback = audioWidget.callback;

                audioWidget.setValue = function(v, skip_callback) {
                    const result = origSetValue?.call(this, v, skip_callback);
                    return result;
                };

                // Update on value change
                audioWidget.callback = function(value) {
                    if (origCallback) {
                        origCallback.call(this, value);
                    }
                };
            }

            // Add select audio file button
            const selectFileBtn = this.addWidget("button", "select_audio", "Select WAV File 🎵", async () => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = ".wav";
                input.style.display = "none";
                document.body.appendChild(input);

                input.onchange = async (e) => {
                    try {
                        if (!e.target.files.length) return;
                        const file = e.target.files[0];

                        // Create form data for upload
                        const formData = new FormData();
                        formData.append("files[]", file);  // Note the "files[]" key
                        formData.append("subfolder", "");
                        formData.append("type", "input");

                        // Use ComfyUI's api.fetchApi
                        const uploadResp = await api.fetchApi("/upload/audio", {
                            method: "POST",
                            body: formData
                        });

                        if (uploadResp.ok) {
                            const data = await uploadResp.json();
                            // Update the audio widget with the file name
                            const audioWidget = this.widgets.find(w => w.name === "audio");
                            if (audioWidget) {
                                audioWidget.value = data.name;  // Use the returned filename
                                if (audioWidget.callback) {
                                    audioWidget.callback(audioWidget.value);
                                }
                                console.log("Audio file uploaded successfully:", data.name);
                            }
                        } else {
                            throw new Error(await uploadResp.text());
                        }
                    } catch (error) {
                        console.error("Error handling file:", error);
                        alert("Error uploading audio file: " + error.message);
                    } finally {
                        document.body.removeChild(input);
                    }
                };

                input.click();
            });

            // Add preview widget
            this.addWidget("preview", "preview", {
                serialize: false,
                size: [256, 256]
            });

            // Ensure proper widget parent references
            if (this.widgets) {
                this.widgets.forEach(w => w.parent = this);
            }

            return result;
        };

        // Add size handling
        nodeType.prototype.onResize = function(size) {
            const minWidth = 400;
            const minHeight = 200;
            size[0] = Math.max(size[0], minWidth);
            size[1] = Math.max(size[1], minHeight);
        };

        // Add execution handling
        nodeType.prototype.onExecuted = function(message) {
            if (message.preview) {
                const previewWidget = this.widgets.find(w => w.name === "preview");
                if (previewWidget) {
                    previewWidget.value = message.preview;
                }
            }
        };
    }
});