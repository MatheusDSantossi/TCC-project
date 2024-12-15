import React from 'react'
import logo from '../../assets/logo.svg';

function Register() {

    const registerEmployee = (userName, email, password, confirmPassword) => {
        
    }
    
    return (
        <div className="flex items-center justify-center min-h-screen p-4">
            <div className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary">
                <div className="flex items-center justify-center mb-5">
                    <img className='w-20 mr-4' src={logo} alt="" />
                    <h2 className="mb-6 text-2xl font-semibold text-center left-40">Register</h2>
                </div>
                <form>
                    <label
                        className="block text-lg font-semibold text-white">
                        Username:
                    </label>
                    <input
                        type="text"
                        name="username"
                        className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                        placeholder='Username'
                    />
                    <label
                        className="block font-semibold text-white text-lm"
                    >
                        Email:
                    </label>
                    <input type="email" name="email"
                        className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                        placeholder='Email'
                    />
                    <label
                        className="block font-semibold text-white text-lm"
                    >
                        Password:
                    </label>
                    <input type="password" name="password"
                        className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                        placeholder='********'
                    />
                    <label
                        className="block font-semibold text-white text-lm"
                    >
                        Confirm Password:
                    </label>
                    <input type="password" name="confirmPassword"
                        className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                        placeholder='********'
                    />
                    <button type="button"
                        className="px-4 py-2 mt-4 font-bold text-black transition duration-200 bg-yellow-400 rounded-lg hover:bg-yellow-300"
                    >
                        Register
                    </button>
                </form>
                <p>Have an account? <a href="/login" className='font-medium'>Login</a></p>
            </div>
        </div>
    )
}

export default Register