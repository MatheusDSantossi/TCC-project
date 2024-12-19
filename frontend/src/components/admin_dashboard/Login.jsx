import React, { useState } from 'react'
import logo from '../../assets/logo.svg';
import { getCSRFToken } from '../.././utils/CSRFToken';

import { Icon } from 'react-icons-kit';
import { eyeOff } from 'react-icons-kit/feather/eyeOff';
import { eye } from 'react-icons-kit/feather/eye'


function Login() {

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [type, setType] = useState('password');
    const [icon, setIcon] = useState(eyeOff);

    const handleClickShowPassword = () => {
        
        if (type==='password'){
            setIcon(eye);
            setType('text')
         } else {
            setIcon(eyeOff)
            setType('password')
         }
    };


    const handleLogin = async (event) => {
        event.preventDefault();

        fetch('http://127.0.0.1:8000/login_employee/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({
                employeeEmail: email.trim(),
                employeePassword: password
            }),
        })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to login');
                }
                return response.json();
            })
            .then((data) => {
                console.log('Logged in successfully', data);
                // Redirect to dashboard page
                window.location.href = '/admin';
            })
            .catch((error) => {
                console.error('Error:', error);
                alert('Failed to login. Please check your credentials.');
            });
    };

    return (
        <div className="flex items-center justify-center min-h-screen p-4">
            <div className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary">
                <div className="flex items-center justify-center mb-5">
                    <img className='w-20 mr-4' src={logo} alt="" />
                    <h2 className="mb-6 text-2xl font-semibold text-center left-40">Login</h2>
                </div>
                <form>
                    <label
                        className="block text-lg font-semibold text-white">
                        Email:
                    </label>
                    <input type="email" placeholder="Email"
                        className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                    />
                    <label
                        className="block font-semibold text-white text-lm"
                    >
                        Password:
                    </label>
                    <div className="flex mb-4">
                        <input
                            type={type}
                            name="password"
                            placeholder="Password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            autoComplete="current-password"
                            className='w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400'
                        />
                        <span className="flex items-center justify-around" onClick={handleClickShowPassword}>
                            <Icon className="absolute mr-10 cursor-pointer hover:text-yellow-400" icon={icon} size={25} />
                        </span>
                    </div>
                    <button type="submit"
                        className="px-4 py-2 mt-4 font-bold text-black transition duration-200 bg-yellow-400 rounded-lg hover:bg-yellow-300"
                        onClick={handleLogin}
                    >
                        Login
                    </button>
                </form>
                <p className='mt-4 text-center text-gray-300'>Don't have an account? <a href="/register" className='font-medium text-yellow-400 hover:underline hover:text-yellow-300'>Register</a></p>
            </div>
        </div>
    )
}

export default Login