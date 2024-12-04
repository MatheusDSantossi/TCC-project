import { useState, useEffect } from 'react';
import logo from '../assets/logo.svg';

function getCSRFToken() {
    const name = 'csrftoken';
    const cookieValue = document.cookie
        .split('; ')
        .find((row) => row.startsWith(name))
        ?.split('=')[1];
    return cookieValue || '';
}

function Capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function Chatbot() {
    const [menuOptions, setMenuOptions] = useState([]);
    const [userInput, setUserInput] = useState('');
    // const [showTextArea, setShowTextArea] = useState(false);
    const [userName, setUserName] = useState('');
    const [isNameEntered, setIsNameEntered] = useState(false);
    const [hasSelectedOption, setHasSelectedOption] = useState(false);
    const [chatMessages, setChatMessages] = useState([
        { sender: 'Hecate', text: `Olá, em que posso lhe ajudar hoje?` }
    ]);

    const [isHecateAnswering, setIsHecateAnswering] = useState(false);

    // Fetch initial menu options from Django backend
    useEffect(() => {
        if (userName) {
            setChatMessages([
                { sender: 'Hecate', text: `Olá, ${Capitalize(userName)}! Como posso lhe ajudar hoje?` }
            ])
        }

        fetchMenuOptions();
    }, [userName]);

    const fetchMenuOptions = () => {


        fetch('http://127.0.0.1:8000/', {
            method: 'GET',

        })
            .then((response) => response.json())
            .then((data) => setMenuOptions(data))
            .catch((error) => console.error('Error fetching menu options:', error));
    };

    // Send a message to the chatbot
    const sendMessage = (message) => {
        const newMessage = { sender: "Você", text: message };
        setChatMessages((prevMessages) => [...prevMessages, newMessage]);

        setIsHecateAnswering(true);

        fetch('http://127.0.0.1:8000/', {
            method: 'POST',
            // credentials: 'include',

            // cache: 'no-cache',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
                // 'X-CSRFToken': '{{ csrfToken }}',
                'X-CSRFToken': getCSRFToken(),
            },
            body: `user_input=${encodeURIComponent(message)}`
        })
            .then((response) => response.json())
            .then((data) => {
                const botResponse = { sender: "Hecate", text: data.response };
                setChatMessages((prevMessages) => [...prevMessages, botResponse]);

            })
            .catch((error) => console.error('Error sending message: ', error));

        setIsHecateAnswering(false);
    };

    // Handle option selection
    const selectOption = (optionId, optionText) => {
        // sendMessage(optionText);
        // setShowTextArea(true); // Show textarea after selecting an option
        setMenuOptions([]); // Hide options after one is selected
        setHasSelectedOption(true); // Indicate an option was selected

        // Fetch the next options based on the selected option ID
        fetch('http://127.0.0.1:8080/', {
            method: 'POST',
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRFToken": getCSRFToken(),
            },
            body: `selected_option_id=${optionId}`
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    console.error('Error fetching next options:', data.error);
                } else {
                    setMenuOptions(data); // Update menu options with the new ones
                }
            })
            .catch((error) => console.error('Error fetching next options:', error));
    };

    const handleNameSubmit = (e) => {
        e.preventDefault();
        if (userName.trim()) {
            setIsNameEntered(true);
        }
    }

    // Handle form submission
    const handleSubmit = (e) => {
        e.preventDefault();
        if (userInput.trim()) {
            setUserInput(''); // clear input field
            sendMessage(userInput);
        }
    };

    return (
        <div className="flex flex-col justify-center items-center w-screen min-h-screen bg-primary p-[72px]">
            <div className="absolute top-4 left-4">
                <img className="w-20" src={logo} alt="Logo" />
            </div>
            {!isNameEntered ? (
                <div className="flex flex-col items-center bg-white shadow-xl rounded-lg p-8">
                    <h2 className="text-lg font-semibold mb-4">Please enter your name:</h2>
                    <form onSubmit={handleNameSubmit} className="w-full flex">
                        <input
                            className="flex-grow p-2 rounded border border-gray-300 focus:outline-none focus:border-blue-500"
                            type="text"
                            placeholder="Your name"
                            value={Capitalize(userName)}
                            onChange={(e) => setUserName(e.target.value)}
                        />
                        <button type="submit" className="ml-2 bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded">
                            Start Chat
                        </button>
                    </form>
                </div>
            ) : (

                < div className="flex flex-col flex-grow w-full max-w-4xl bg-white shadow-xl rounded-lg overflow-hidden">
                    <div className="flex flex-col flex-grow h-0 p-4 overflow-auto">
                        {/* Chat Message */}
                        {chatMessages.map((message, index) => (

                            <div
                                key={index}
                                className={`flex w-full mt-2 space-x-3 max-w-xs ${message.sender === 'Você' ? 'ml-auto justify-end' : ''
                                    }`}
                            >
                                <div>
                                    <div className='flex-shrink-0 h-10 w-10 rounded-full bg-gray-100'></div>
                                    <span className='text-black font-semibold'>{message.sender == 'Você' ? `${Capitalize(userName)}` : "Hecate"}</span>
                                </div>

                                <div>
                                    <div
                                        className={`${message.sender === 'Você' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-black'} p-3 rounded-lg`}
                                    >
                                        <p className="text-sm">{message.text}</p>
                                    </div>
                                    <span className="text-sx text-gray-50 leading-none">2 min ago</span>
                                </div>
                            </div>
                        ))}


                        {/* Options container */}
                        {!hasSelectedOption && (
                            <div className='options-container mb-4 grid gap-2'>
                                {menuOptions.map((option, index) => (
                                    <button
                                        key={index}
                                        className='bg-yellow-400 hover:bg-yellow-300 text-black font-bold py-2 px-4 rounded-lg transition duration-200'
                                        onClick={() => {
                                            selectOption(option.id, option.menu_option_description, 1);
                                        }}
                                    >
                                        {option.menu_option_description}
                                    </button>
                                ))}
                            </div>

                        )}

                        {isHecateAnswering &&
                            <span className='font-bold text-2xl'>Hecate está pensando...</span>}
                        {/* Input field */}
                        {hasSelectedOption && (

                            <div className="absolute bottom-0 left-0 w-full bg-gray-300">

                                <form
                                    onSubmit={handleSubmit}
                                    className="flex items-center px-4 py-3">

                                    <input
                                        className="flex-grow h-10 px-3 border border-gray-400 rounded-l-md focus:outline-none focus:ring focus:ring-yellow-500" type="text"
                                        placeholder="Type your message…" onSubmit={handleSubmit}
                                        onChange={(e) => setUserInput(e.target.value)}
                                        value={userInput}
                                    />

                                    <button type="submit" className="h-10 px-6 ml-2 bg-blue-500 hover:bg-blue-600 text-white rounded-r-md">
                                        Send
                                    </button>

                                </form>
                            </div >
                            // </div>
                        )}
                    </div>
                </div >
            )}
        </div>
    )
}

export default Chatbot;
