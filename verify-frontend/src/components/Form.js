import React, { useState, useEffect } from "react";
import { PhotoIcon } from "@heroicons/react/24/solid";

const Form = () => {
    const [users, setUsers] = useState([]);
    const [selectedUser, setSelectedUser] = useState(null);
    const [genuineImagePreview, setGenuineImagePreview] = useState(null);
    const [forgedImagePreview, setForgedImagePreview] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    // Fetch users from the API
    useEffect(() => {
        const fetchUsers = async () => {
            const response = await fetch("http://127.0.0.1:5000/get_users");
            const data = await response.json();
            setUsers(data.data);
            console.log(users);
        };
        fetchUsers();
    }, []);

    const handleUserSelection = (event) => {
        const userId = event.target.value;
        const user = users.find((user) => user.id === userId);
        setSelectedUser(user);
        if (user) {
            setGenuineImagePreview(user.signature_image);
        }
    };

    const handleForgedFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setForgedImagePreview(URL.createObjectURL(file));
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        setIsLoading(true);

        // Construct FormData
        const formData = new FormData();
        // Append files here, ensure you have references to the file inputs
        // Example: formData.append('forgedImage', forgedImageFile);

        // Submit formData to your backend
        console.log("Submitting form...");

        // Reset after submission for demonstration
        setIsLoading(false);
        setGenuineImagePreview(null);
        setForgedImagePreview(null);
    };

    return (
        <div className="mx-auto max-w-4xl p-8">
            <div>
                <label
                    htmlFor="user-select"
                    className="block text-sm font-medium text-gray-900"
                >
                    Select User:
                </label>
                <div className="mt-1 relative">
                    <select
                        id="user-select"
                        className="appearance-none block w-full px-3 py-2 border border-gray-300 text-base rounded-md shadow-sm placeholder-gray-500 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                        onChange={handleUserSelection}
                        value={selectedUser ? selectedUser.id : ""}
                    >
                        <option value="">Select a user</option>
                        {users.map((user) => (
                            <option key={user.id} value={user.id}>
                                {user.name}
                            </option>
                        ))}
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700">
                        <svg
                            className="h-4 w-4 fill-current"
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                        >
                            <path d="M5.292 7.293a1 1 0 011.414 0L10 10.586l3.294-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" />
                        </svg>
                    </div>
                </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-8 mt-4">
                {/* Genuine Signature Display */}
                <div>
                    <label className="block text-sm font-medium leading-6 text-gray-900">
                        Genuine Signature
                    </label>
                    <div className="mt-1 flex justify-center rounded-md border-2 border-dashed border-gray-300 p-6">
                        {genuineImagePreview ? (
                            <img
                                src={genuineImagePreview}
                                alt="Genuine Signature Preview"
                                className="max-h-60"
                            />
                        ) : (
                            <div className="space-y-1 text-center">
                                <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
                                <p className="text-sm text-gray-600">
                                    No signature loaded
                                </p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Forged Signature Upload */}
                <div>
                    <label className="block text-sm font-medium leading-6 text-gray-900">
                        Upload Forged Signature
                    </label>
                    <div className="mt-1 flex justify-center rounded-md border-2 border-dashed border-gray-300 p-6">
                        {forgedImagePreview ? (
                            <img
                                src={forgedImagePreview}
                                alt="Forged Signature Preview"
                                className="max-h-60"
                            />
                        ) : (
                            <div className="space-y-1 text-center">
                                <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
                                <div className="flex text-sm text-gray-600">
                                    <label
                                        htmlFor="forged-signature"
                                        className="relative cursor-pointer rounded-md bg-white text-indigo-600 focus-within:outline-none focus-within:ring-2 focus-within:ring-indigo-500 focus-within:ring-offset-2 hover:text-indigo-500"
                                    >
                                        <span>Upload a file</span>
                                        <input
                                            id="forged-signature"
                                            name="forgedSignature"
                                            type="file"
                                            className="sr-only"
                                            onChange={handleForgedFileChange}
                                        />
                                    </label>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Submit Button */}
                <div>
                    <button
                        type="submit"
                        className="text-sm font-semibold leading-6 text-white bg-indigo-600 border border-transparent rounded-md shadow-sm px-4 py-2 w-full transition duration-150 ease-in-out hover:bg-indigo-500 focus:outline-none focus:border-indigo-700 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                    >
                        {isLoading ? "Processing..." : "Verify"}
                    </button>
                </div>
            </form>
        </div>
    );
};

export default Form;
