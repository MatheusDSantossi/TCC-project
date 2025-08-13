import { useState, useEffect, useRef } from "react";
import logo from "../assets/logo.svg";
import hecatePFP from "../assets/hecate_pfp.jpg";
import test_image from "../assets/test_image.jpg";
import Typewriter from "./Typewriter";
import AxiosInstance from "./AxiosInstance";
import { useNavigate, useParams } from "react-router-dom";
import { HashLoader } from "react-spinners";
import ExceptionMainComponent from "../exceptions/ExceptionMainComponent";
import { ExceptionTypes } from "../exceptions/ExceptionVariantsTypes";

function Capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function Chatbot() {
  const navigate = useNavigate();

  // States
  const [userInput, setUserInput] = useState("");
  const [userName, setUserName] = useState("");
  const [isNameEntered, setIsNameEntered] = useState(false);

  const [menuOptions, setMenuOptions] = useState([]);
  const [hasSelectedOption, setHasSelectedOption] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [isHecateAnswering, setIsHecateAnswering] = useState(false);
  const [isTypingDone, setIsTypingDone] = useState(false);
  const [pendingMenuOptions, setPendingMenuOptions] = useState(null);
  const [isTextRelatedTo, setIsTextRelatedTo] = useState("default");
  const chatContainerRef = useRef(null);
  const [
    showMenuOptionsAndInputAfterHecateAnswer,
    setShowMenuOptionsAndInputAfterHecateAnswer,
  ] = useState(false);

  const [showPreviousMenuButton, setShowPreviousMenuButton] = useState(false);

  const [exceptionsData, setExceptionsData] = useState(null);

  const { businessSlug } = useParams(); // e.g. "supermarket-santos"
  const localBusinessSlug = localStorage.getItem("BusinessSlug") || null;

  const finalBusinessSlug = businessSlug || localBusinessSlug;

  const url = `/${finalBusinessSlug}`;

  // useEffect(() => {
  //   console.log("Chatbot mounted")
  // }, [chatMessages])

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
    setExceptionsData(null)

  }, [chatMessages, menuOptions]);

  const fetchMenuOptions = () => {
    setShowPreviousMenuButton(false);
    AxiosInstance.get(`menu-options/get/`)
      // .then((response) => setMenuOptions(response.data))
      .then((response) => {
        console.log("Fetching menu options return: ", response.data);
        setPendingMenuOptions(response.data);
      })
      .catch((error) => {
        console.error("Error fetching menu options:", error);
        setExceptionsData({
          text: "Error fetching menu options",
          errorMessage: error,
          variant: ExceptionTypes.ERROR,
        });
        // ExceptionMainComponent("Error fetching menu options", error, ExceptionTypes.ERROR);
      });
  };

  // Fetch initial menu options from Django backend
  useEffect(() => {
    if (!userName) {
      setChatMessages([
        {
          sender: "Hecate",
          text: `Olá me chamo Hecate! Como você se chama?`,
          timestamp: new Date().toLocaleTimeString("pt-BR", {
            hour: "2-digit",
            minute: "2-digit",
          }), // Static timestamp
        },
      ]);
    }

    fetchMenuOptions();
  }, [userName]);

  useEffect(() => {
    if (menuOptions.length <= 0) {
      setShowMenuOptionsAndInputAfterHecateAnswer(true);
    }
  }, [menuOptions]);

  const handleSessionEnding = (e) => {
    // setIsHecateAnswering(true);
    // setIsTypingDone(false);
    const clientText = e.target.textContent;
    const clientMessage = {
      sender: "Você",
      text: clientText,
      timestamp: new Date().toLocaleTimeString("pt-BR", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
    const hecateMessage = {
      sender: "Hecate",
      text: "Estou a disposição se precisar volte a falar comigo. :)",
      timestamp: new Date().toLocaleTimeString("pt-BR", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
    setChatMessages((prevMessages) => [...prevMessages, clientMessage]);
    setShowMenuOptionsAndInputAfterHecateAnswer(false);

    setTimeout(() => {
      setChatMessages((prevMessages) => [...prevMessages, hecateMessage]);
    }, 3000);
    setTimeout(() => {
      navigate(url);
    }, 8000);
  };

  // Send a message to the chatbot
  const sendMessage = async (message, showMessage = true) => {
    setIsHecateAnswering(true);
    setIsTypingDone(false);

    const newMessage = {
      sender: "Você",
      text: message,
      timestamp: new Date().toLocaleTimeString("pt-BR", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };

    if (showMessage) {
      setChatMessages((prevMessages) => [...prevMessages, newMessage]);
    }

    console.log("Sending message at:", new Date().toISOString());

    try {
      console.log("Sending message...");

      // Await the Axios POST request
      const response = await AxiosInstance.post(`chatbot/post/`, {
        user_input: message,
        text_related_to: isTextRelatedTo,
        business_slug: finalBusinessSlug,
      });

      setIsHecateAnswering(false);
      const botResponse = {
        sender: "Hecate",
        text: response.data.response,
        timestamp: new Date().toLocaleTimeString("pt-BR", {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };
      setChatMessages((prevMessages) => [...prevMessages, botResponse]);

      // showingptions();

      // setHasSelectedOption(false);
      // fetchMenuOptions();

      // const botResponseAfterFirstAnswer = { sender: "Hecate", text: "Posso lhe ajudar em algo mais?"};
      // setChatMessages((prevMessages) => [...prevMessages, botResponseAfterFirstAnswer]);
    } catch (error) {
      console.error("Error sending message: ", error);
    }
  };
  // Handle typing completion
  const handleTypingEnd = () => {
    console.log("Got typing completion");
    if (pendingMenuOptions) {
      console.log("Setting pending menu options");
      console.log("Pending menu options: ", pendingMenuOptions);
      setMenuOptions(pendingMenuOptions);
      setPendingMenuOptions(null);
    }
    setIsTypingDone(true); // Enable options or next actions
  };

  // Handle option selection
  const selectOption = (optionId, optionText) => {
    const userMessage = {
      sender: "Você",
      text: optionText,
      timestamp: new Date().toLocaleTimeString("pt-BR", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    };
    setChatMessages((prevMessages) => [...prevMessages, userMessage]);

    setMenuOptions([]); // Clear options menu
    setIsTypingDone(false);

    // Fetch the next options based on the selected option ID
    AxiosInstance.post(`menu-options/post/`, {
      selected_option_id: optionId,
    })
      .then((response) => {
        console.log(`Full response: ${response}`);
        console.log(`menu options data: ${response.data}`);
        console.log(`option Text: ${optionText}`);

        console.log(
          "showMenuOptionsAfterHecateAnswer: ",
          showMenuOptionsAndInputAfterHecateAnswer
        );
        console.log(
          "Showmenuoptions: ",
          showMenuOptionsAndInputAfterHecateAnswer
        );
        console.log("isTypingDone: ", isTypingDone);
        console.log("isHecateAnswering: ", isHecateAnswering);
        console.log("MenuOptions lenght: ", menuOptions.length);

        if (response.data.message === "No more options available") {
          let answer = "Digite sua dúvida.";
          if (optionText === "Digitar o nome do produto") {
            answer = "Digite o nome do produto.";
          } else if (optionText === "Digitar a seção do produto") {
            setIsTextRelatedTo("section");
            answer = "Digite a seção do produto. (Ex: Seção 1)";
          } else if (optionText === "Digitar a marca do produto") {
            setIsTextRelatedTo("brand");
            answer = "Digite a marca do produto. (Ex: Vitamassa)";
          } else if (optionText === "Digitar alguma descrição do produto") {
            answer = "Digite alguma descrição do produto.";
          } else if (optionText === "Digitar o código do produto") {
            answer = "Digite o código do produto.";
          } else if (
            optionText === "Digitar alguma descrição do produto ou sinônimo"
          ) {
            answer = "Digite a descrição ou sinônimo do produto.";
          } else if (
            optionText === "Nome do gerente" ||
            optionText === "Quais são as promoções do dia?" ||
            optionText == "Lista de categorias disponíveis"
          ) {
            answer = "";
            sendMessage(optionText, false);
          }

          const chatbotAnswer = {
            sender: "Hecate",
            text: answer,
            timestamp: new Date().toLocaleTimeString("pt-BR", {
              hour: "2-digit",
              minute: "2-digit",
            }),
          };
          if (chatbotAnswer.text.length > 0) {
            setChatMessages((prevMessages) => [...prevMessages, chatbotAnswer]);
          }
          setHasSelectedOption(true);
        } else {
          // get this message based on the user menu choice
          // TODO: Get the text based on the menuOption, so I think I have to do this in the view python
          // because I think it'll be easier.
          const botConfirmation = {
            sender: "Hecate",
            text: "Entendi. Selecione uma das opções abaixo:",
            timestamp: new Date().toLocaleTimeString("pt-BR", {
              hour: "2-digit",
              minute: "2-digit",
            }),
          };
          setChatMessages((prevMessages) => [...prevMessages, botConfirmation]);

          console.log(`menu options data: ${response.data}`);

          setPendingMenuOptions(response.data);

          handleTypingEnd();

          setShowPreviousMenuButton(true);
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
        text: `${newUserName}`,
        timestamp: new Date().toLocaleTimeString("pt-BR", {
          hour: "2-digit",
          minute: "2-digit",
        }),
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
          text: `Prazer em conhecê-lo, ${newUserName}! Como posso lhe ajudar hoje? Selecione uma das opções abaixo:`,
          timestamp: new Date().toLocaleTimeString("pt-BR", {
            hour: "2-digit",
            minute: "2-digit",
          }),
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
    // setIsTypingDone(false);
    if (e.key === "Enter" && userInput.trim()) {
      handleNameSubmit();
    }
  };

  const showingMenuOptions = (e) => {
    e.preventDefault();
    const lastHecateMessage = chatMessages
      .filter((msg) => msg.sender === "Hecate")
      .pop();

    console.log("Last message hecate: " + lastHecateMessage.text);
    if (
      lastHecateMessage.text !== "Entendi. Selecione uma das opções abaixo:"
    ) {
      const botConfirmation = {
        sender: "Hecate",
        text: "Selecione uma das opções abaixo:",
        timestamp: new Date().toLocaleTimeString("pt-BR", {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };
      setChatMessages((prevMessages) => [...prevMessages, botConfirmation]);
      setShowMenuOptionsAndInputAfterHecateAnswer(false);
      setTimeout(() => {
        fetchMenuOptions();
      }, 3000);
    } else {
      setShowMenuOptionsAndInputAfterHecateAnswer(false);
      fetchMenuOptions();
    }
  };

  return (
    <div className="flex flex-col justify-center items-center w-screen min-h-screen bg-primary p-[72px] relative">
      <div className="absolute top-4 left-4">
        <img className="w-20" src={logo} alt="Logo" />
      </div>

      <div className="flex flex-col flex-grow w-full max-w-4xl overflow-hidden bg-white rounded-lg shadow-xl">
        {exceptionsData && (
          <ExceptionMainComponent
            text={exceptionsData.text}
            errorMessage={exceptionsData.errorMessage}
            variant={exceptionsData.variant}
          />
        )}
        <div
          className="flex flex-col flex-grow h-0 p-4 overflow-auto text-left"
          ref={chatContainerRef}
        >
          {/* Chat Message */}
          {chatMessages.map((message, index) => (
            <div
              key={index}
              className={`flex w-full mt-2 space-x-3 max-w-xs ${
                message.sender === "Você" ? "ml-auto justify-end text-left" : ""
              }`}
            >
              <div>
                <div className="flex-shrink-0 w-10 h-10 rounded-full">
                  {message.sender == "Hecate" ? (
                    <img
                      src={hecatePFP}
                      alt="Profile picture of Hecate"
                      className="rounded-full"
                    />
                  ) : (
                    <img
                      src={test_image}
                      alt="Profile picture of user"
                      className="rounded-full"
                    />
                  )}
                </div>
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
                      ? "bg-black text-white"
                      : "bg-gray-200 text-black"
                  } p-4 border-gray-200 rounded-e-xl rounded-es-xl mt-5 text-lg leading-relaxed`}
                >
                  {/* <div
                  className={`${
                    message.sender === "Você"
                      ? "bg-black text-white"
                      : "bg-gray-200 text-black"
                  } p-3 rounded-lg`}
                > */}
                  {message.sender == "Você" ? (
                    <p className="text-lg">{message.text}</p>
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
                {/* Dynamic Time Display */}
                <span className="font-medium leading-none text-sx text-primary">
                  {message.timestamp}
                </span>
              </div>
            </div>
          ))}
          {!isNameEntered && isTypingDone && (
            <div className="absolute bottom-3 left-1/2 transform -translate-x-1/2 w-full max-w-[600px] z-50 px-4">
              <div className="flex items-center w-full mt-4 space-x-2">
                <input
                  className="flex-grow px-4 py-3 text-lg rounded-md bg-input focus:outline-none focus:ring-2 focus:ring-yellow-500"
                  type="text"
                  placeholder="Digite seu nome"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyDown={handleKeyPress}
                  autoFocus
                />
                <button
                  className="flex-shrink-0 px-4 py-2 ml-2 font-semibold text-black bg-yellow-400 rounded cursor-pointer hover:bg-yellow-300 whitespace-nowrap"
                  onClick={handleNameSubmit}
                >
                  Enviar
                </button>
              </div>
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
                    className="px-4 py-2 text-lg font-semibold text-black transition duration-200 bg-yellow-400 rounded-lg cursor-pointer hover:bg-yellow-300"
                    onClick={() => {
                      selectOption(
                        option.menu_extended_id,
                        option.description,
                        1
                      );
                    }}
                  >
                    {option.description}
                  </button>
                ))}
              </div>
              {/* )} */}
            </div>
          )}
          {/* Put as the chatbot is saying that instead of a simple span */}
          {isHecateAnswering && (
            // <span className="text-2xl font-bold text-primary">
            //   Estou processando. Aguarde um instante.
            // </span>
            <span className="flex text-2xl font-bold text-primary">
              Estou processando. Aguarde um instante.
              <div className="mr-8"></div>
              <HashLoader
                color="#012921"
                size={30}
                aria-label="Loading Spinner"
              />
            </span>
          )}

          {showMenuOptionsAndInputAfterHecateAnswer &&
            hasSelectedOption &&
            !isHecateAnswering &&
            menuOptions <= 0 &&
            isTypingDone && (
              <div className="flex justify-between">
                <button
                  className="h-10 px-6 ml-2 font-semibold text-black bg-yellow-400 cursor-pointer hover:bg-yellow-300 rounded-r-md"
                  onClick={showingMenuOptions}
                >
                  {/* onClick={showingMenuOptions()} */}
                  Voltar para a lista de opções
                </button>
                <button
                  className="h-10 px-6 ml-2 font-semibold text-black bg-yellow-400 cursor-pointer hover:bg-yellow-300 rounded-r-md"
                  onClick={handleSessionEnding}
                >
                  {/* onClick={showingMenuOptions()} */}
                  Encerrar atendimento
                </button>
              </div>
            )}

          {/* Return to the previous menu button */}
          {showPreviousMenuButton &&
            isTypingDone &&
            !isHecateAnswering &&
            menuOptions.length >= 1 && (
              <button
                className="h-10 px-6 ml-2 font-semibold text-black bg-yellow-400 hover:bg-yellow-300 rounded-r-md"
                onClick={showingMenuOptions}
              >
                {/* onClick={handlePreviousMenu()} */}
                Voltar para o menu anterior
              </button>
            )}

          {/* Input field */}
          {showMenuOptionsAndInputAfterHecateAnswer &&
            hasSelectedOption &&
            !isHecateAnswering &&
            menuOptions <= 0 &&
            isTypingDone && (
              <div className="absolute bottom-3 left-1/2 transform -translate-x-1/2 w-full max-w-[600px] px-4">
                <form
                  onSubmit={handleSubmit}
                  className="flex items-center w-full mt-4 space-x-2"
                >
                  <input
                    className="flex-grow px-4 py-3 text-lg border rounded-md bg-input focus:outline-none focus:ring-2 focus:ring-yellow-500"
                    type="text"
                    placeholder="Digite sua mensagem…"
                    onSubmit={handleSubmit}
                    onChange={(e) => setUserInput(e.target.value)}
                    value={userInput}
                    autoFocus
                  />

                  <button
                    type="submit"
                    className="h-10 px-6 ml-2 font-semibold text-black bg-yellow-400 rounded hover:bg-yellow-300"
                  >
                    Enviar
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
