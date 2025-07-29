# ShelfSmart
ShelfSmart is an AI-powered data-as-a-service platform that enables businesses to turn raw data into clear, actionable insights with intuitive analytics and visualization tools.

## Frontend (React + Vite)

**Setup and run:**

```bash
npm install
npm run dev
```

The frontend will be at:
[http://localhost:5173](http://localhost:5173)

## Backend (FastAPI)

**Setup and run:**

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API docs will be at:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Database (PostgreSQL)

1. **Ensure PostgreSQL is running.**

2. **Create the database (if not exists):**

```bash
createdb -U postgres shelfsmart_db
````

3. **Set up `.env`:**

```env
DATABASE_URL=postgresql+psycopg2://<username>:<password>@<host>:<port>/shelfsmart_db
SECRET_KEY=your_super_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

* Replace `<password>` (e.g., `000000`) and `<port>` (e.g., `5432`) with **your own PostgreSQL password and port**.

4. **Tables are auto-created when the FastAPI app runs.**












# Welcome to your Lovable project

## Project info

**URL**: https://lovable.dev/projects/c66b39d8-f02b-4249-92ba-dc293f5c1766

## How can I edit this code?

There are several ways of editing your application.

**Use Lovable**

Simply visit the [Lovable Project](https://lovable.dev/projects/c66b39d8-f02b-4249-92ba-dc293f5c1766) and start prompting.

Changes made via Lovable will be committed automatically to this repo.

**Use your preferred IDE**

If you want to work locally using your own IDE, you can clone this repo and push changes. Pushed changes will also be reflected in Lovable.

The only requirement is having Node.js & npm installed - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)

Follow these steps:

```sh
# Step 1: Clone the repository using the project's Git URL.
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory.
cd <YOUR_PROJECT_NAME>

# Step 3: Install the necessary dependencies.
npm i

# Step 4: Start the development server with auto-reloading and an instant preview.
npm run dev
```

**Edit a file directly in GitHub**

- Navigate to the desired file(s).
- Click the "Edit" button (pencil icon) at the top right of the file view.
- Make your changes and commit the changes.

**Use GitHub Codespaces**

- Navigate to the main page of your repository.
- Click on the "Code" button (green button) near the top right.
- Select the "Codespaces" tab.
- Click on "New codespace" to launch a new Codespace environment.
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/c66b39d8-f02b-4249-92ba-dc293f5c1766) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/tips-tricks/custom-domain#step-by-step-guide)
