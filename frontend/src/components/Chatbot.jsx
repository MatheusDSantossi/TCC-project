import { useState, useEffect } from "react";
import logo from "../assets/logo.svg";
import Typewriter from "./Typewriter";
import AxiosInstance from "./AxiosInstance";
// import ReactTyped

function Capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function Chatbot() {
  const [menuOptions, setMenuOptions] = useState([]);
  const [userInput, setUserInput] = useState("");
  // const [showTextArea, setShowTextArea] = useState(false);
  const [userName, setUserName] = useState("");
  const [isNameEntered, setIsNameEntered] = useState(false);
  const [hasSelectedOption, setHasSelectedOption] = useState(false);
  // const [chatMessages, setChatMessages] = useState([
  //   { sender: "Hecate", text: `Olá, me chamo Hecate! Qual é o seu nome?` },
  // ]);
  const [chatMessages, setChatMessages] = useState([]);

  const [isHecateAnswering, setIsHecateAnswering] = useState(false);

  const [isTypingDone, setIsTypingDone] = useState(false);
  const [pendingMenuOptions, setPendingMenuOptions] = useState(null);

  // Fetch initial menu options from Django backend
  useEffect(() => {
    if (!userName) {
      setChatMessages([
        {
          sender: "Hecate",
          text: `Olá me chamo Hecate! Como você se chama?`,
        },
      ]);
    }

    fetchMenuOptions();
  }, [userName]);

  const fetchMenuOptions = () => {
    AxiosInstance.get(`menu-options/get`)
      // .then((response) => setMenuOptions(response.data))
      .then((response) => setPendingMenuOptions(response.data))
      .catch((error) => console.error("Error fetching menu options:", error));
  };

  // Send a message to the chatbot
  const sendMessage = async (message) => {
    setIsHecateAnswering(true);
    setIsTypingDone(false);

    const newMessage = { sender: "Você", text: message };
    setChatMessages((prevMessages) => [...prevMessages, newMessage]);

    console.log("Sending message at:", new Date().toISOString());

    try {
      console.log("Sending message...");
      // Await the Axios POST request
      const response = await AxiosInstance.post(`chatbot/post/`, {
        user_input: message,
      });
      // TODO: Even with this timeout the return is immediately

      setIsHecateAnswering(false);
      const botResponse = { sender: "Hecate", text: response.data.response };
      setChatMessages((prevMessages) => [...prevMessages, botResponse]);

      // console.log("Received response at:", new Date().toISOString());
      // console.log("Received response: ", response);

      // console.log("The chatbot return: ", response);
    } catch (error) {
      console.error("Error sending message: ", error);
    }
  };
  // Handle typing completion
  const handleTypingEnd = () => {
    console.log("Got typing completion");
    if (pendingMenuOptions) {
      console.log("Setting pending menu options");
      console.log(pendingMenuOptions);
      setMenuOptions(pendingMenuOptions);
      setPendingMenuOptions(null);
    }
    setIsTypingDone(true); // Enable options or next actions
  };

  // Handle option selection
  const selectOption = (optionId, optionText) => {
    const userMessage = { sender: "Você", text: optionText };
    setChatMessages((prevMessages) => [...prevMessages, userMessage]);

    // // get this message based on the user menu choice
    // const botConfirmation = {
    //   sender: "Hecate",
    //   text: "Entendi. Selecione uma das opções:",
    // };
    // setChatMessages((prevMessages) => [...prevMessages, botConfirmation]);

    setMenuOptions([]); // Clear options menu
    setIsTypingDone(false);

    // Fetch the next options based on the selected option ID
    AxiosInstance.post(`menu-options/post/`, {
      selected_option_id: optionId,
    })
      .then((response) => {
        console.log(`Full response: ${response}`);
        console.log(`menu options data: ${response.data}`);

        if (response.data === "No more options available") {
          const chatbotAnswer = {
            sender: "Hecate",
            text: "Digite sua dúvida.",
          };

          setChatMessages((prevMessages) => [...prevMessages, chatbotAnswer]);
          setHasSelectedOption(true);
        } else {
          // get this message based on the user menu choice
          // TODO: Get the text based on the menuOption, so I think I have to do this in the view python
          // because I think it'll be easier.
          const botConfirmation = {
            sender: "Hecate",
            text: "Entendi. Selecione uma das opções:",
          };
          setChatMessages((prevMessages) => [...prevMessages, botConfirmation]);

          console.log(`menu options data: ${response.data}`);

          setPendingMenuOptions(response.data);
          // };

          handleTypingEnd();
        }
      })
      .catch((error) => console.error("Error fetching next options:", error));
  };

  // const handleNameSubmit = (e) => {
  const handleNameSubmit = () => {
    const value = userInput.replace(/[^a-zA-Z\s]/g, "");
    console.log("User input: " + userInput);
    console.log(
      "User input value formatted without numbers or special characters: " +
        value
    );
    setIsTypingDone(false);

    if (userInput.trim()) {
      setUserName(value);
      const newUserName = value;
      setIsNameEntered(true);
      console.log(newUserName);

      // Addding the user's message to the chat
      const userNameMessage = {
        sender: "Você",
        text: `Me chamo ${newUserName}`,
      };

      setChatMessages((prevMessages) => {
        const updatedMessages = [...prevMessages, userNameMessage];
        console.log(
          "Chat messages adding userNameMessage answer:",
          updatedMessages
        );
        return updatedMessages;
      });

      // Add the chatbot's response
      setTimeout(() => {
        const botResponse = {
          sender: "Hecate",
          text: `Prazer em conhecê-lo, ${newUserName}! Como posso lhe ajudar hoje?`,
        };
        setChatMessages((prevMessages) => {
          const updatedMessages = [...prevMessages, botResponse];
          console.log("Chat messages after botResponse:", updatedMessages);
          return updatedMessages;
        });
      }, 500);

      setUserInput("");
    }
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    setIsTypingDone(false);
    if (userInput.trim()) {
      setUserInput(""); // clear input field
      sendMessage(userInput);
    }
  };

  // handle key press
  const handleKeyPress = (e) => {
    setIsTypingDone(false);
    if (e.key === "Enter" && userInput.trim()) {
      handleNameSubmit();
    }
  };

  return (
    <div className="flex flex-col justify-center items-center w-screen min-h-screen bg-primary p-[72px] relative">
      <div className="absolute top-4 left-4">
        <img className="w-20" src={logo} alt="Logo" />
      </div>

      <div className="flex flex-col flex-grow w-full max-w-4xl overflow-hidden bg-white rounded-lg shadow-xl">
        <div className="flex flex-col flex-grow h-0 p-4 overflow-auto">
          {/* Chat Message */}
          {chatMessages.map((message, index) => (
            <div
              key={index}
              className={`flex w-full mt-2 space-x-3 max-w-xs ${
                message.sender === "Você" ? "ml-auto justify-end" : ""
              }`}
            >
              <div>
                <div className="flex-shrink-0 w-10 h-10 bg-gray-100 rounded-full"></div>
                <span className="font-semibold text-black">
                  {message.sender == "Você"
                    ? `${Capitalize(userName)}`
                    : "Hecate"}
                </span>
              </div>

              <div>
                <div
                  className={`${
                    message.sender === "Você"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-200 text-black"
                  } p-3 rounded-lg`}
                >
                  {message.sender == "Você" ? (
                    <p className="text-sm">{message.text}</p>
                  ) : (
                    <Typewriter
                      className="text-sm"
                      text={`${message.text}`}
                      delay={30}
                      onTypingEnd={
                        index === chatMessages.length - 1
                          ? handleTypingEnd
                          : null
                      }
                    />
                  )}
                </div>
                <span className="font-medium leading-none text-sx text-primary">
                  2 min ago
                </span>
              </div>
            </div>
          ))}
          {/* {!isNameEntered && isTypingDone && ( */}
          {!isNameEntered && (
            <div className="flex items-center mt-4">
              <input
                className="w-full px-4 py-3 border border-gray-400 rounded-md focus:outline-none focus:ring focus:ring-yellow-500"
                type="text"
                placeholder="Digite seu nome?"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                onKeyDown={handleKeyPress}
              />
              <button
                className="flex-shrink-0 px-4 py-2 ml-2 text-white bg-yellow-400 rounded-md hover:bg-yellow-300"
                onClick={handleNameSubmit}
              >
                Enviar
              </button>
            </div>
          )}
          {/* {isNameEntered && isTypingDone && menuOptions.length > 0 && ( */}
          {isNameEntered && isTypingDone && menuOptions.length > 0 && (
            <div>
              {/* Options container */}
              {/* {!hasSelectedOption && ( */}
              {/* {menuOptions.length > 0 && ( */}
              <div className="grid gap-2 mb-4 options-container">
                {menuOptions.map((option, index) => (
                  <button
                    key={index}
                    className="px-4 py-2 font-bold text-black transition duration-200 bg-yellow-400 rounded-lg hover:bg-yellow-300"
                    onClick={() => {
                      selectOption(
                        option.menu_extended_id,
                        option.menu_option_description,
                        1
                      );
                    }}
                  >
                    {option.menu_option_description}
                  </button>
                ))}
              </div>
              {/* )} */}
            </div>
          )}
          {/* Put as the chatbot is saying that instead of a simple span */}
          {isHecateAnswering && (
            <span className="text-2xl font-bold text-primary">
              Aguarde. Em breve responderei!
            </span>
          )}
          {/* Input field */}
          {hasSelectedOption && (
            <div className="absolute bottom-0 left-0 w-full bg-gray-300">
              <form
                onSubmit={handleSubmit}
                className="flex items-center px-4 py-3"
              >
                <input
                  className="flex-grow h-10 px-3 border border-gray-400 rounded-l-md focus:outline-none focus:ring focus:ring-yellow-500"
                  type="text"
                  placeholder="Type your message…"
                  onSubmit={handleSubmit}
                  onChange={(e) => setUserInput(e.target.value)}
                  value={userInput}
                />

                <button
                  type="submit"
                  className="h-10 px-6 ml-2 text-white bg-blue-500 hover:bg-blue-600 rounded-r-md"
                >
                  Send
                </button>
              </form>
            </div>
            // </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Chatbot;
