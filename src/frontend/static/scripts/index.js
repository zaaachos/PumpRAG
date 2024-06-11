document.addEventListener("DOMContentLoaded", () => {
  const chatBox = document.getElementById("chat-box");
  const messageInput = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");
  const ragSwitch = document.getElementById("rag-switch");
  let isRagEnabled = ragSwitch.checked;
  // Add event listener to handle changes to the switch
  ragSwitch.addEventListener("change", () => {
    isRagEnabled = ragSwitch.checked;
    console.log(isRagEnabled ? "RAG is enabled" : "RAG is disabled");
  });

  // Initialize the connection of user with virtual assistant via WebSockets
  const ws = new WebSocket("ws://localhost:8001/chat");

  // When connected to WebSocket server, log a message (for debugging purposes)
  ws.onopen = () => {
    console.log("Connected to WebSocket server");
  };

  // Function to send message
  const sendMessage = () => {
    const message = messageInput.value.trim();
    if (message !== "") {
      appendMessage(message, "sent");
      const messageData = JSON.stringify({ message, isRagEnabled });
      ws.send(messageData);
      // ws.send(message);
      messageInput.value = "";
      appendMessage("", "received");
    }
  };

  // Add event listener for send button
  sendButton.addEventListener("click", sendMessage);

  // Add event listener for Enter key press in the input box
  messageInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      sendMessage();
    }
  });

  // Function to update the chat box with the new message
  ws.onmessage = (event) => {
    const message = event.data;
    console.log(message);
    updateChat(message);
  };

  // Function to update the existing message in the chat box
  function updateChat(message) {
    const messageElements = chatBox.getElementsByClassName("message");
    // Update the text content of the last message element
    const lastMessageElement = messageElements[messageElements.length - 1];
    lastMessageElement.querySelector(".content .text").textContent += message;

    // Set maximum scroll
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  // Function to append a new DIV element for each message
  function appendMessage(message, type) {
    const messageContainer = document.createElement("div");
    messageContainer.classList.add("message-container", type);

    const messageElement = document.createElement("div");
    messageElement.classList.add("message", type);

    const content = document.createElement("div");
    content.classList.add("content");

    const text = document.createElement("div");
    text.classList.add("text");
    text.textContent = message;

    const timestamp = document.createElement("div");
    timestamp.classList.add("timestamp");
    timestamp.textContent = new Date().toLocaleTimeString();
    const icon = document.createElement("div");
    icon.classList.add("icon");
    content.appendChild(timestamp);

    content.appendChild(text);
    if (type == "received") {
      messageElement.appendChild(icon);
      messageElement.appendChild(content);
    } else {
      messageElement.appendChild(content);
      messageElement.appendChild(icon);
    }

    messageContainer.appendChild(messageElement);
    chatBox.appendChild(messageContainer);

    // Set maximum scroll
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});
