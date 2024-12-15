import React from 'react'
import logo from '../../assets/logo.svg';

function Login() {
    return (
        <div className="flex items-center justify-center min-h-screen p-4">
            <div className="w-full max-w-lg p-8 rounded-lg shadow-xl bg-primary">
                <div className="flex items-center justify-center mb-5">
                    <img className='w-20 mr-4' src={logo} alt="" />
                    <h2 className="mb-6 text-2xl font-semibold text-center left-40">Login</h2>
                </div>
                <form>
                    <input type="text" placeholder="Username" 
                    className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                    />
                    <input type="password" placeholder="Password" 
                    className="w-full p-2 mt-1 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
                    />
                    <button type="submit"
                    className="px-4 py-2 mt-4 font-bold text-black transition duration-200 bg-yellow-400 rounded-lg hover:bg-yellow-300"
                    >
                        Login
                    </button>
                </form>
                <p>Don't have an account? <a href="/register" className='font-medium'>Register</a></p>
            </div>
        </div>
    )
}

export default Login