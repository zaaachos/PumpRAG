document.addEventListener("DOMContentLoaded", () => {
  const chatBox = document.getElementById("chat-box");
  const messageInput = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");

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
      ws.send(message);
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
    icon.innerHTML = type === "sent" ? "&#128100;" : "&#128187;"; // User icon and bot icon

    content.appendChild(text);
    content.appendChild(timestamp);
    messageElement.appendChild(content);
    messageElement.appendChild(icon);
    messageContainer.appendChild(messageElement);
    chatBox.appendChild(messageContainer);

    // Set maximum scroll
    chatBox.scrollTop = chatBox.scrollHeight;
  }
});
